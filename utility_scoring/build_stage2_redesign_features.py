#!/usr/bin/env python3
"""Experiment 2 of STAGE2_REDESIGN_SPEC: build the revised 11-feature matrix.

Feature families (SPEC sections 5-10):

  A  matching        s_sem, s_lex
  B  raw route       p_dense, i_dense, p_bm25, i_bm25, p_graph, i_graph
  C  candidate       log_len
  D  anchor context  s_max_d8, s_mean_d8

The central experimental boundary (SPEC sections 6, 7, 20) is enforced here by
construction: the scorer sees RAW Dense/BM25/Graph route evidence, and never
the Dense/RRF2/RRF3 FUSED entry ordering, which is reserved for the selection
layer's Residual Prior.  Gate G2 proves this empirically rather than by
assertion: features are built independently from two different entry pools and
every shared (query, candidate) row must come out bit-identical, which is only
possible if no entry-order information reached the feature vector.

Deterministic redundancy (novelty = 1 - s_max) is removed per SPEC section 9.
No PCA, clustering, or unsupervised reduction is used anywhere (SPEC sections
3, 12, 30): every feature is an explicitly engineered retrieval-time signal.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEY = "stage2_redesign_features_rawtext"

sys.path.insert(0, str(ROOT))
from evidence_selection import run_selection_action_space_repair as repair  # noqa: E402

try:
    import configuration as project_config
    from evaluation.community_reply_auxiliary import now, sha256_file
    from evaluation.judgment_completeness import complete_utility_v2_rows
    from candidate_pool.run_dense_semantic_drift_rescue_audit import (
        _build_idf, _cosine, _lexical_diagnostics, _load_embeddings, _tokenize,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT))
    import configuration as project_config
    from evaluation.community_reply_auxiliary import now, sha256_file
    from evaluation.judgment_completeness import complete_utility_v2_rows
    from candidate_pool.run_dense_semantic_drift_rescue_audit import (
        _build_idf, _cosine, _lexical_diagnostics, _load_embeddings, _tokenize,
    )

CONTINUOUS = ("s_sem", "s_lex", "p_dense", "p_bm25", "p_graph",
              "log_len", "s_max_d8", "s_mean_d8")
INDICATORS = ("i_dense", "i_bm25", "i_graph")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


def _read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percentile(rank: int | None, depth: int) -> float:
    """SPEC section 6: p_r = 1 - (j-1)/(R-1); absent from the route -> 0.0.

    One consistent convention for all three routes, unlike the frozen
    eight-feature contract's (101-rank)/100, which hard-codes a depth-100
    normaliser and therefore cannot be shared by routes of differing depth.
    """
    if rank is None:
        return 0.0
    if not 1 <= rank <= depth:
        raise ValueError(f"rank {rank} outside the route depth {depth}")
    return 1.0 - (rank - 1) / (depth - 1)


def _route_ranks(cfg: dict[str, Any]) -> dict[str, dict[str, dict[str, int]]]:
    """query -> candidate -> rank, per raw route.  No fusion is read here."""
    dense: dict[str, dict[str, int]] = defaultdict(dict)
    for row in _read_jsonl(cfg["dense_memberships"]):
        if str(row["backend"]) != str(cfg["backend"]):
            continue
        dense[str(row["query_id"])][str(row["comment_id"])] = int(row["rank"])
    bm25: dict[str, dict[str, int]] = defaultdict(dict)
    for row in pq.read_table(cfg["bm25_rankings"]).to_pylist():
        bm25[str(row["query_id"])][str(row["comment_id"])] = int(row["rank"])
    graph: dict[str, dict[str, int]] = defaultdict(dict)
    for row in _read_jsonl(cfg["graph_memberships"]):
        graph[str(row["query_id"])][str(row["candidate_id"])] = int(row["graph_rank"])
    return {"D_dense": dict(dense), "B_bm25": dict(bm25), "G_graph": dict(graph)}


def _build(pool: dict[str, list[str]], *, cfg, ranks, depths, anchors,
           candidate_vectors, query_vectors, corpus_text, query_text,
           idf) -> dict[tuple[str, str], dict[str, float]]:
    """The registered feature columns for every (query, candidate) in the pool.

    Nothing in this function can see an entry ordering: it consumes only raw
    per-route ranks, texts, and the fixed Dense Top-8 anchor.
    """
    names = list(map(str, cfg["feature_names"]))
    out: dict[tuple[str, str], dict[str, float]] = {}
    for qid in sorted(pool):
        anchor_vectors = [candidate_vectors[cid] for cid in anchors[qid]]
        for cid in pool[qid]:
            vector = candidate_vectors[cid]
            anchor_similarities = [_cosine(vector, a) for a in anchor_vectors]
            lexical = _lexical_diagnostics(query_text[qid], corpus_text[cid], idf)
            row = {
                "s_sem": _cosine(query_vectors[qid], vector),
                "s_lex": float(lexical["idf_weighted_lexical_overlap"]),
                "log_len": math.log1p(len(_tokenize(corpus_text[cid]))),
                "s_max_d8": max(anchor_similarities),
                "s_mean_d8": statistics.fmean(anchor_similarities),
            }
            # clean7D transfer needs no BM25 or graph feature source. Calculate
            # only registered route columns; the full eleven-column path is
            # numerically unchanged and keeps its configured column order.
            for route, suffix in (("D_dense", "dense"), ("B_bm25", "bm25"), ("G_graph", "graph")):
                if f"p_{suffix}" in names or f"i_{suffix}" in names:
                    rank = ranks[route][qid].get(cid)
                    row[f"p_{suffix}"] = _percentile(rank, depths[route])
                    row[f"i_{suffix}"] = float(rank is None)
            out[(qid, cid)] = {name: row[name] for name in names}
    return out


def run(config_key: str = CONFIG_KEY, output_dir: Path | None = None) -> dict[str, Any]:
    cfg = dict(project_config.load()[config_key])
    if cfg.get("allow_external_calls") or cfg.get("allow_frozen_test"):
        raise ValueError("feature construction is local and development-only")
    for key in ("output_dir", "corpus", "queries", "dense_memberships",
                "graph_memberships", "bm25_rankings", "rrf3_ordered_ids",
                "utility_registry"):
        cfg[key] = _resolve(cfg[key])
        if key != "output_dir" and "test" in str(cfg[key]).lower():
            raise ValueError(f"G5: {key} resolves to a path resembling test scope")
    destination = Path(output_dir or cfg["output_dir"]).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")

    depths = {k: int(v) for k, v in dict(cfg["route_depths"]).items()}
    names = list(map(str, cfg["feature_names"]))
    ranks = _route_ranks(cfg)

    # ---- entry pools (identities only; their ORDER never reaches a feature)
    dense_pool = {
        qid: [cid for cid, _ in sorted(rows.items(), key=lambda kv: kv[1])]
        [: int(cfg["pool_depth"])]
        for qid, rows in ranks["D_dense"].items()
    }
    anchors = {qid: ids[: int(cfg["anchor_size"])] for qid, ids in dense_pool.items()}
    rrf3_pool = {
        str(row["query_id"]): [str(c) for c in row["ordered_ids"]]
        for row in _read_jsonl(cfg["rrf3_ordered_ids"])
    }
    qids = sorted(dense_pool)
    if len(qids) != int(cfg["expected_queries"]) or set(rrf3_pool) != set(qids):
        raise ValueError("G5: query cohort is not the frozen Dev300")
    universe = {qid: sorted(set(dense_pool[qid]) | set(rrf3_pool[qid])) for qid in qids}

    # ---- texts, embeddings, IDF ------------------------------------------
    source_raw = project_config.load()[str(cfg["source_config_key"])]
    paths = repair._resolve_inputs(source_raw)
    backend = str(cfg["backend"])
    (candidate_vectors, query_vectors, corpus_text, query_text,
     _corpus_array, _corpus_ids) = _load_embeddings(
        corpus_path=Path(cfg["corpus"]),
        corpus_embeddings_path=paths["embeddings"][backend]["corpus"],
        queries_path=Path(cfg["queries"]),
        query_embeddings_path=paths["embeddings"][backend]["query"],
        qids=set(qids),
    )
    idf, idf_manifest = _build_idf(corpus_text.values())

    # ---- G1: features are built BEFORE any utility label is loaded -------
    features = _build(
        universe, cfg=cfg, ranks=ranks, depths=depths, anchors=anchors,
        candidate_vectors=candidate_vectors, query_vectors=query_vectors,
        corpus_text=corpus_text, query_text=query_text, idf=idf,
    )

    # ---- G2: entry-invariance.  Rebuild from two different entry pools and
    # require bit-identical values on every shared pair.  If any fused entry
    # ordering had leaked into a feature, these would differ.
    from_dense = _build(
        dense_pool, cfg=cfg, ranks=ranks, depths=depths, anchors=anchors,
        candidate_vectors=candidate_vectors, query_vectors=query_vectors,
        corpus_text=corpus_text, query_text=query_text, idf=idf,
    )
    from_rrf3 = _build(
        rrf3_pool, cfg=cfg, ranks=ranks, depths=depths, anchors=anchors,
        candidate_vectors=candidate_vectors, query_vectors=query_vectors,
        corpus_text=corpus_text, query_text=query_text, idf=idf,
    )
    shared = sorted(set(from_dense) & set(from_rrf3))
    invariance_violations = [
        pair for pair in shared
        if any(from_dense[pair][n] != from_rrf3[pair][n] for n in names)
        or any(from_dense[pair][n] != features[pair][n] for n in names)
    ]
    if invariance_violations:
        raise AssertionError(
            f"G2 FAILED: {len(invariance_violations)} shared pairs differ between "
            "entry pools; an entry ordering has leaked into the features"
        )

    # ---- G4: reproducible ordering ---------------------------------------
    ordered_pairs = sorted(features)
    if ordered_pairs != sorted(set(ordered_pairs)):
        raise AssertionError("G4: duplicate (query, candidate) rows")

    # ---- now, and only now, utility is loaded (coverage check only) ------
    _complete, registry = complete_utility_v2_rows(
        _read_jsonl(cfg["utility_registry"])
    )
    if len(registry) != int(cfg["expected_registry_rows"]):
        raise ValueError("utility registry identity changed")
    unlabelled = [pair for pair in ordered_pairs if pair not in registry]

    rows = [
        {"query_id": qid, "candidate_id": cid,
         "in_dense_pool": bool(cid in set(dense_pool[qid])),
         "in_rrf3_pool": bool(cid in set(rrf3_pool[qid])),
         **{n: float(features[(qid, cid)][n]) for n in names},
         "utility": float(registry[(qid, cid)]["utility"])
         if (qid, cid) in registry else None}
        for qid, cid in ordered_pairs
    ]

    # ---- feature statistics and within-training-fold correlations --------
    matrix = np.asarray([[row[n] for n in names] for row in rows], dtype=np.float64)
    stats_rows = [
        {"position": index + 1, "feature": name,
         "family": ("A_matching" if name in ("s_sem", "s_lex") else
                    "B_raw_route" if name in ("p_dense", "i_dense", "p_bm25",
                                              "i_bm25", "p_graph", "i_graph") else
                    "C_candidate" if name == "log_len" else "D_anchor_context"),
         "min": float(matrix[:, index].min()),
         "median": float(np.median(matrix[:, index])),
         "max": float(matrix[:, index].max()),
         "mean": float(matrix[:, index].mean()),
         "missing_rate": 0.0,
         "zero_rate": float((matrix[:, index] == 0.0).mean())}
        for index, name in enumerate(names)
    ]

    splits = json.loads(
        Path(paths["split_manifest"]).read_text(encoding="utf-8")
    )["rows"]
    continuous_index = [names.index(n) for n in CONTINUOUS]
    per_fold = []
    for split in splits:
        train = set(map(str, split["train_query_ids"]))
        mask = np.asarray([row["query_id"] in train for row in rows])
        block = matrix[np.ix_(mask, continuous_index)]
        per_fold.append(np.corrcoef(block, rowvar=False))
    mean_corr = np.mean(np.stack(per_fold), axis=0)
    corr_rows = [
        {"feature": CONTINUOUS[i],
         **{CONTINUOUS[j]: float(mean_corr[i, j]) for j in range(len(CONTINUOUS))}}
        for i in range(len(CONTINUOUS))
    ]
    flagged = [
        {"feature_a": CONTINUOUS[i], "feature_b": CONTINUOUS[j],
         "mean_abs_rho_across_training_folds": float(abs(mean_corr[i, j]))}
        for i in range(len(CONTINUOUS)) for j in range(i + 1, len(CONTINUOUS))
        if abs(mean_corr[i, j]) > 0.9
    ]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as tmp:
        out = Path(tmp)
        pq.write_table(pa.Table.from_pylist(rows),
                       out / "stage2_features_11.parquet", compression="zstd")
        _write_csv(out / "feature_statistics.csv", stats_rows)
        _write_csv(out / "feature_correlations_training_folds.csv", corr_rows)
        manifest = {
            "schema": "stage2-redesign-features-v1",
            "version": str(cfg["version"]),
            "status": "COMPLETE",
            "created_utc": now(),
            "experiment": "SPEC Experiment 2 - revised feature matrix",
            "backend": backend,
            "feature_names": names,
            "feature_families": {
                "A_matching": ["s_sem", "s_lex"],
                "B_raw_route_evidence": ["p_dense", "i_dense", "p_bm25",
                                         "i_bm25", "p_graph", "i_graph"],
                "C_candidate_characteristic": ["log_len"],
                "D_relation_to_fixed_dense_top8": ["s_max_d8", "s_mean_d8"],
            },
            "percentile_convention": "p_r = 1 - (rank-1)/(depth-1); absent -> 0.0",
            "route_depths": depths,
            "anchor": "fixed Dense Top-8, identical across all entry conditions",
            "removed_from_frozen_contract": {
                "candidate_novelty_relative_to_d8":
                    "deterministically redundant, = 1 - s_max_d8 (SPEC section 9)",
            },
            "gates": {
                "G1_features_built_before_utility_loaded": True,
                "G2_entry_invariance_shared_pairs": len(shared),
                "G2_entry_invariance_violations": len(invariance_violations),
                "G3_route_id_categorical_shortcut": False,
                "G3_disclosure": (
                    "i_dense/i_bm25/i_graph are route-membership proxies by "
                    "design (SPEC section 6); no one-hot route identity is added"
                ),
                "G4_rows_sorted_and_unique": True,
                "G5_cohort_is_dev300_only": True,
                "G5_test_paths_rejected": True,
                "unlabelled_pairs": len(unlabelled),
                "label_coverage": 1.0 - len(unlabelled) / len(rows),
            },
            "counts": {
                "queries": len(qids),
                "rows": len(rows),
                "dense_pool_rows": sum(len(v) for v in dense_pool.values()),
                "rrf3_pool_rows": sum(len(v) for v in rrf3_pool.values()),
                "union_rows": len(rows),
                "features": len(names),
            },
            "idf_manifest": idf_manifest,
            "redundancy_diagnostics": {
                "threshold": 0.9,
                "flagged_pairs": flagged,
                "note": ("mean Pearson correlation across the 25 outer training "
                         "folds, continuous features only; development data only "
                         "(SPEC section 13).  Flagging is diagnostic: nothing is "
                         "removed automatically"),
            },
            "boundaries": {
                "external_requests_made": 0,
                "frozen_test_read": False,
                "fusion_entry_rank_used_as_feature": False,
                "utility_used_as_feature": False,
                "unsupervised_reduction_used": False,
                "model_training_performed": False,
            },
            "outputs": {
                name: sha256_file(out / name) for name in (
                    "stage2_features_11.parquet", "feature_statistics.csv",
                    "feature_correlations_training_folds.csv",
                )
            },
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        Path(out).rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-key", default=CONFIG_KEY)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = run(args.config_key, args.output_dir)
    print(json.dumps({"counts": manifest["counts"], "gates": manifest["gates"],
                      "redundancy": manifest["redundancy_diagnostics"]},
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
