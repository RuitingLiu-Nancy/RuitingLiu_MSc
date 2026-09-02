"""Multi-source fusion retrieval: semantic + bm25 + multi-hop.

Two fusion modes (both standard in the IR literature):

  RRF (Reciprocal Rank Fusion, rank-based, zero-shot):
      score(d) = Σ_s  w_s / (k0 + rank_s(d))
    - Uses only RANKS, so per-source score scales don't matter.
    - The 1/(k0+rank) shape gives noisy multi-hop hits (which land at high rank)
      a SMALL weight, so they can't crowd out strong semantic hits. This is
      exactly what tames multi-hop's drift/over-retrieval (StepChain, RobustGraphRAG)
      while still letting multi-hop contribute the EXCLUSIVE gold it alone reaches.
    - Strong, robust default for small ensembles / no training data.

  CC (Convex Combination, score-based, needs a tiny bit of tuning):
      score(d) = Σ_s  w_s * minmax_norm(raw_score_s(d)),  Σ w_s = 1
    - Per-source scores are min-max normalised to [0,1] before weighting.
    - Literature (Bruch et al., ACM TOIS 2023) shows CC can beat RRF when you
      have a small dev set to tune the weights — you have held-out gold for that.

References: Reciprocal Rank Fusion (Cormack 2009); An Analysis of Fusion
Functions for Hybrid Retrieval (Bruch et al., ACM TOIS 2023).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _as_scored(ranked):
    """[(comment_id, raw_score)] from an arm's ranked dicts (keep input order)."""
    out = []
    for c in ranked or []:
        cid = str(c.get("comment_id", c.get("cid", "")))
        if cid:
            out.append((cid, float(c.get("score", 0.0)), c))
    return out


def rrf_fuse(source_runs: dict, weights: dict, k0: int = 60, k: int = 8):
    """source_runs: {name: ranked_list}. Returns fused ranked list of dicts."""
    score = rrf_scores(source_runs, weights=weights, k0=k0)
    meta = {}
    for name, ranked in source_runs.items():
        for cid, _raw, c in _as_scored(ranked):
            meta.setdefault(cid, c)
    ordered = sorted(score.items(), key=lambda x: -x[1])[:k]
    return [dict(meta[cid], score=round(s, 5), fused="rrf") for cid, s in ordered]


def rrf_scores(source_runs: dict, weights: dict | None = None,
               k0: int = 60) -> dict[str, float]:
    """Return untruncated canonical RRF scores for a set of ranked runs.

    Pool builders use the complete score map because they need to fill a
    label-blind union to a target size rather than return a retrieval top-k.
    """
    weights = weights or {}
    score = defaultdict(float)
    for name, ranked in source_runs.items():
        weight = weights.get(name, 1.0)
        for rank, (cid, _raw, _row) in enumerate(_as_scored(ranked), 1):
            score[cid] += weight / (k0 + rank)
    return dict(score)


def _minmax(vals):
    if not vals:
        return {}
    lo, hi = min(vals.values()), max(vals.values())
    if hi <= lo:
        return {kk: 1.0 for kk in vals}
    return {kk: (v - lo) / (hi - lo) for kk, v in vals.items()}


def _zscore(vals):
    """Query-local z-score normalisation with a deterministic constant case."""
    if not vals:
        return {}
    ordered = np.asarray(list(vals.values()), dtype=float)
    std = float(ordered.std())
    if std <= 0.0:
        return {key: 0.0 for key in vals}
    mean = float(ordered.mean())
    return {key: (float(value) - mean) / std for key, value in vals.items()}


def _rank_percentile(vals):
    """Map larger query-local scores to larger [0, 1] rank percentiles."""
    if not vals:
        return {}
    ordered = sorted(vals, key=lambda key: (float(vals[key]), key))
    if len(ordered) == 1:
        return {ordered[0]: 1.0}
    return {key: rank / (len(ordered) - 1) for rank, key in enumerate(ordered)}


def normalize_scores(vals: dict[str, float], method: str = "minmax") -> dict[str, float]:
    """Canonical query-wise score normalisation used by controlled CC arms."""
    if method == "minmax":
        return _minmax(vals)
    if method == "zscore":
        return _zscore(vals)
    if method == "rank_percentile":
        return _rank_percentile(vals)
    raise ValueError(f"unknown score normalization: {method}")


def cc_scores(source_runs: dict, weights: dict,
              normalization: str = "minmax") -> dict[str, float]:
    """Return untruncated convex-combination scores for a source union."""
    norm_per_source = {}
    for name, ranked in source_runs.items():
        raw = {cid: sc for cid, sc, _ in _as_scored(ranked)}
        norm_per_source[name] = normalize_scores(raw, normalization)
    total_w = sum(weights.get(name, 0.0) for name in source_runs) or 1.0
    score = defaultdict(float)
    for name in source_runs:
        weight = weights.get(name, 0.0) / total_w
        for cid, value in norm_per_source[name].items():
            score[cid] += weight * value
    return dict(score)


def cc_fuse(source_runs: dict, weights: dict, k: int = 8,
            normalization: str = "minmax"):
    """Convex combination of min-max normalised per-source scores."""
    meta = {}
    for name, ranked in source_runs.items():
        for cid, sc, c in _as_scored(ranked):
            meta.setdefault(cid, c)
    score = cc_scores(source_runs, weights, normalization=normalization)
    ordered = sorted(score.items(), key=lambda x: -x[1])[:k]
    return [dict(meta[cid], score=round(s, 5), fused="cc") for cid, s in ordered]


def fuse(source_runs: dict, *, mode: str = "rrf", weights: dict | None = None,
         k0: int = 60, k: int = 8, normalization: str = "minmax"):
    weights = weights or {}
    if mode == "cc":
        return cc_fuse(source_runs, weights, k=k, normalization=normalization)
    return rrf_fuse(source_runs, weights, k0=k0, k=k)


# --------------------------------------------------------------------------- #
#  Tune CC weights on held-out gold (grid search over a simplex of 3 weights).
#  Lightweight, no ML deps. Returns best weights + the metric it optimised.
# --------------------------------------------------------------------------- #
def tune_cc_weights(eval_rows, build_runs, k=8, step=0.1, metric="ndcg"):
    """eval_rows: [{query, gold:set}]. build_runs(query)-> {name: ranked_list}.
    Grid-search convex weights for (semantic, bm25, multihop)."""
    import math

    def ndcg(ids, gold):
        dcg = sum(1 / math.log2(i + 2) for i, c in enumerate(ids[:k]) if c in gold)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
        return dcg / ideal if ideal else 0.0

    def recall(ids, gold):
        return len(set(ids[:k]) & gold) / len(gold) if gold else 0.0

    score_fn = ndcg if metric == "ndcg" else recall
    names = ("semantic", "bm25", "multihop")
    # precompute runs once
    cached = [(r["gold"], build_runs(r["query"])) for r in eval_rows if r.get("gold")]
    best, best_w = -1, None
    grid = [i * step for i in range(int(1 / step) + 1)]
    for a in grid:
        for b in grid:
            c = round(1 - a - b, 5)
            if c < -1e-9:
                continue
            w = {"semantic": a, "bm25": b, "multihop": c}
            vals = []
            for gold, runs in cached:
                fused = cc_fuse(runs, w, k=k)
                ids = [str(x.get("comment_id")) for x in fused]
                vals.append(score_fn(ids, gold))
            avg = sum(vals) / len(vals) if vals else 0
            if avg > best:
                best, best_w = avg, w
    return best_w, round(best, 4)
