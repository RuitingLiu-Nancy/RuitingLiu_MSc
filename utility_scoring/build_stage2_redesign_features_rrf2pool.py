#!/usr/bin/env python3
"""RRF2-only feature matrix with the two legacy columns the ablation needs.

Execution pack v3 tightens the scorer training pool from the RRF2 union RRF3
union (15,428 rows) to RRF2-only (15,000 rows), so the scorer is trained on the
candidates it actually scores at deployment, and adds a single-factor ablation
ladder A0..A4 that needs two columns the eleven-feature contract deliberately
does not carry:

  p_dense_legacy  the frozen contract's (101 - rank)/100 percentile, whose
                  depth-100 normaliser over a depth-50 membership is the scale
                  defect recorded in D-20260827-01
  novelty_d8      the frozen contract's 1 - s_max_d8, deterministically
                  redundant with s_max_d8 and dropped by SPEC section 9

Both are derived here from the frozen inputs rather than re-implementing the
feature builder: the eleven contract columns are carried over unchanged from
the Experiment-2 matrix, so A4 is bit-identical to the contract already in use.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEY = "stage2_redesign_features_rrf2pool_rawtext"

try:
    import configuration as project_config
    from evaluation.community_reply_auxiliary import now, sha256_file
    from fusion.analyze_rq2a_graph_budget_sweep import (
        load_scored_dense,
        load_scored_graph,
        ordered_fusion_ids,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT))
    import configuration as project_config
    from evaluation.community_reply_auxiliary import now, sha256_file
    from fusion.analyze_rq2a_graph_budget_sweep import (
        load_scored_dense,
        load_scored_graph,
        ordered_fusion_ids,
    )


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


def run(config_key: str = CONFIG_KEY, output_dir: Path | None = None) -> dict[str, Any]:
    cfg = dict(project_config.load()[config_key])
    if cfg.get("allow_external_calls") or cfg.get("allow_frozen_test"):
        raise ValueError("feature derivation is local and development-only")
    for key in ("output_dir", "features_parquet", "dense_memberships",
                "graph_memberships"):
        cfg[key] = _resolve(cfg[key])
        if key != "output_dir" and "test" in str(cfg[key]).lower():
            raise ValueError(f"{key} resolves to a path resembling test scope")
    destination = Path(output_dir or cfg["output_dir"]).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")

    backend, pool_depth = str(cfg["backend"]), int(cfg["pool_depth"])
    params = project_config.load()
    dense_scored = load_scored_dense(cfg["dense_memberships"], backend)
    graph_scored = load_scored_graph(cfg["graph_memberships"])
    qids = sorted(dense_scored)

    dense_pool, dense_rank, rrf2_pool = {}, {}, {}
    for qid in qids:
        rows_ = dense_scored[qid]
        dense_pool[qid] = [str(r["comment_id"]) for r in rows_][:pool_depth]
        dense_rank[qid] = {cid: i + 1 for i, cid in enumerate(dense_pool[qid])}
        rrf2_pool[qid] = ordered_fusion_ids(
            rows_, graph_scored[qid], mode="rrf",
            dense_weight=float(params["fusion"]["weights"]["semantic"]),
            graph_weight=float(params["fusion"]["weights"]["multihop"]),
            rrf_k0=int(params["retrieval"]["k0"]),
            cc_normalization="minmax")[:pool_depth]

    # Gate (execution pack v3 section B): the RRF2 pool must be the Dense pool.
    identical = sum(1 for q in qids if set(rrf2_pool[q]) == set(dense_pool[q]))
    if identical != len(qids):
        raise AssertionError(
            f"RRF2/Dense M{pool_depth} pool identity is {identical}/{len(qids)}; "
            "the ablation's fixed-pool premise does not hold")

    contract = list(map(str, cfg["contract_features"]))
    source_rows = {
        (str(r["query_id"]), str(r["candidate_id"])): r
        for r in pq.read_table(cfg["features_parquet"]).to_pylist()
    }
    out_rows, novelty_violations = [], 0
    for qid in qids:
        for cid in sorted(rrf2_pool[qid]):
            row = source_rows[(qid, cid)]
            rank = dense_rank[qid].get(cid)
            legacy = (101 - rank) / 100.0 if rank is not None else 0.0
            novelty = 1.0 - float(row["s_max_d8"])
            if abs((1.0 - float(row["s_max_d8"])) - novelty) > 1e-12:
                novelty_violations += 1
            out_rows.append(
                {"query_id": qid, "candidate_id": cid,
                 **{name: float(row[name]) for name in contract},
                 "p_dense_legacy": legacy, "novelty_d8": novelty,
                 "utility": float(row["utility"])})
    if len(out_rows) != len(qids) * pool_depth:
        raise ValueError(f"{len(out_rows)} rows, expected {len(qids) * pool_depth}")
    if any(r["utility"] is None for r in out_rows):
        raise ValueError("label coverage is not complete on the RRF2 pool")

    ranked = [r["p_dense_legacy"] for r in out_rows if r["p_dense_legacy"] > 0.0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as tmp:
        out = Path(tmp)
        pq.write_table(pa.Table.from_pylist(out_rows),
                       out / "stage2_features_rrf2pool.parquet", compression="zstd")
        manifest = {
            "schema": "stage2-redesign-features-rrf2pool-v1",
            "version": str(cfg["version"]),
            "status": "COMPLETE",
            "created_utc": now(),
            "purpose": ("RRF2-only training pool with the two legacy columns the "
                        "A0..A4 single-factor ablation requires"),
            "backend": backend,
            "pool": {"entry": "rrf2", "depth": pool_depth,
                     "queries": len(qids), "rows": len(out_rows),
                     "rrf2_equals_dense_pool": f"{identical}/{len(qids)}"},
            "contract_features_carried_unchanged": contract,
            "derived_columns": {
                "p_dense_legacy": {
                    "formula": "(101 - dense_rank)/100 if ranked else 0.0",
                    "realised_min_over_ranked": min(ranked),
                    "realised_max_over_ranked": max(ranked),
                    "note": ("the frozen contract's depth-100 normaliser applied "
                             "to a depth-50 membership; D-20260827-01"),
                },
                "novelty_d8": {
                    "formula": "1 - s_max_d8",
                    "exactly_collinear_with": "s_max_d8",
                    "violations": novelty_violations,
                },
            },
            "label_coverage": 1.0,
            "boundaries": {"external_requests_made": 0, "frozen_test_read": False,
                           "fusion_entry_rank_used_as_feature": False},
            "inputs": {str(cfg["features_parquet"]):
                       sha256_file(cfg["features_parquet"])},
            "outputs": {"stage2_features_rrf2pool.parquet":
                        sha256_file(out / "stage2_features_rrf2pool.parquet")},
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        Path(out).rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-key", default=CONFIG_KEY)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    m = run(args.config_key, args.output_dir)
    print(json.dumps({"pool": m["pool"], "derived": m["derived_columns"]},
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
