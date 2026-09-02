"""Score external retrieval routes on the frozen route-balanced utility pool.

This is a pooled-judgement evaluation: rankings are projected onto the same
pre-existing candidate pool, while the route must retrieve at least ``k`` pool
candidates within its exported depth. It never creates or reads test labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.external_fusion_utility_rerank import _read_admin
from evaluation.ir_metrics import graded_ndcg_at
from data_preparation.sampling.sample_human_annotation_candidates import _load_cards
from evaluation.score_multihop_retrieval import _load_pred, paired_bootstrap_delta


def _bare(value: str) -> str:
    return Path(str(value)).stem


def score_routes(
    admin_pool: Path,
    study_db: Path,
    heldout: Path,
    predictions: dict[str, Path],
    *,
    k: int = 8,
    reasonable_threshold: float = 4.0,
) -> dict:
    if any("test" in path.name.lower() for path in
           [admin_pool, study_db, heldout, *predictions.values()]):
        raise ValueError("frozen test artifacts are forbidden")
    admin = _read_admin(admin_pool)
    cards = _load_cards(study_db, heldout)
    gains = {(_bare(row["query_id"]), str(row["comment_id"])): float(row["utility"])
             for row in cards}
    pool: dict[str, set[str]] = defaultdict(set)
    for row in admin:
        qid = _bare(row.get("canonical_query_id") or row["query_id"])
        pool[qid].add(str(row["comment_id"]))
    missing_labels = sorted((qid, cid) for qid, ids in pool.items() for cid in ids
                            if (qid, cid) not in gains)
    if missing_labels:
        raise RuntimeError(f"utility pool has {len(missing_labels)} unrated candidates")

    summaries, per_query = {}, {}
    for name, path in predictions.items():
        raw = {_bare(qid): values for qid, values in _load_pred(path).items()}
        ndcg, utility_mean, reasonable, coverage = {}, [], [], []
        for qid, candidates in pool.items():
            projected = [cid for cid in raw.get(qid, []) if cid in candidates]
            if len(projected) < k:
                raise RuntimeError(
                    f"{name} retrieves only {len(projected)} judged candidates for {qid}; "
                    f"need at least k={k} for a fair projected ranking")
            query_gains = {cid: gains[(qid, cid)] for cid in candidates}
            ndcg[qid] = float(graded_ndcg_at(projected, query_gains, k))
            top = projected[:k]
            utility_mean.append(float(np.mean([query_gains[cid] for cid in top])))
            reasonable.append(float(np.mean([
                query_gains[cid] >= reasonable_threshold for cid in top])))
            coverage.append(len(projected))
        per_query[name] = ndcg
        summaries[name] = {
            f"graded_utility_ndcg@{k}": float(np.mean(list(ndcg.values()))),
            f"utility_mean@{k}": float(np.mean(utility_mean)),
            f"reasonable_share@{k}": float(np.mean(reasonable)),
            "mean_judged_candidates_retrieved": float(np.mean(coverage)),
            "queries": len(ndcg),
        }
    paired = []
    names = list(predictions)
    for left_idx in range(len(names)):
        for right_idx in range(left_idx + 1, len(names)):
            left, right = names[left_idx], names[right_idx]
            delta = paired_bootstrap_delta(per_query[left], per_query[right])
            paired.append({
                "left": left, "right": right,
                "metric": f"graded_utility_ndcg@{k}", **(delta or {}),
            })
    return {
        "protocol": "fixed route-balanced pooled utility evaluation v1",
        "test_split_used": False,
        "admin_pool": str(admin_pool),
        "study_db": str(study_db),
        "heldout": str(heldout),
        "pool_queries": len(pool),
        "pool_candidates": sum(map(len, pool.values())),
        "k": k,
        "reasonable_threshold": reasonable_threshold,
        "systems": summaries,
        "paired": paired,
        "claim_guard": "LLM utility-v2 is silver; human calibration remains required.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admin-pool", type=Path, required=True)
    ap.add_argument("--study-db", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--pred", action="append", required=True, help="name=path")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    predictions = {}
    for item in args.pred:
        name, sep, value = item.partition("=")
        if not sep or not name or not value:
            raise ValueError(f"invalid --pred: {item}")
        predictions[name] = Path(value)
    result = score_routes(
        args.admin_pool, args.study_db, args.heldout, predictions, k=args.k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
