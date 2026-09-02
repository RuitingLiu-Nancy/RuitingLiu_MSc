#!/usr/bin/env python3
"""Analyse the frozen RQ2a Graph-budget sweep after utility-v2 completion.

This is a deterministic local analysis.  It reuses the Chapter 5 Oracle U@8
and whole-query bootstrap implementation, never calls an external provider,
never reads Test200, and never changes a frozen candidate pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.consolidate_rq2_complete_experiment_data import (  # noqa: E402
    rq2_bootstrap_indices,
    rq2_load_rankings,
    rq2_oracle,
    rq2_paired_summary,
    rq2_read_jsonl,
    rq2_sha256,
)
from configuration import params  # noqa: E402
from fusion.ranking import (  # noqa: E402
    cc_scores,
    normalize_scores,
    rrf_scores,
)
from shared.io_utils import write_csv_rows  # noqa: E402


FINAL_K = 8
BOOTSTRAP_SEED = 20260805
BOOTSTRAP_DRAWS = 5000
BACKENDS = ("minilm", "e5")
DENSE_DEPTHS = (8, 12, 20, 50)
GRAPH_BUDGETS = (4, 8, 12, 20, 50)

STAGE1_METHODS = (
    ("dense_only", "Dense only"),
    ("dense_70_graph_30", "Dense:Graph 70:30"),
    ("dense_50_graph_50", "Dense:Graph 50:50"),
    ("dense_30_graph_70", "Dense:Graph 30:70"),
    ("graph_only", "Graph only"),
    ("rrf", "RRF fusion"),
    ("cc", "CC fusion"),
)


def stable_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(list(values))


def load_utility(path: Path) -> dict[tuple[str, str], float]:
    rows = rq2_read_jsonl(path)
    utility: dict[tuple[str, str], float] = {}
    for row in rows:
        pair = (str(row["query_id"]), str(row["comment_id"]))
        if pair in utility:
            raise ValueError(f"duplicate utility identity: {pair}")
        utility[pair] = float(row["utility"])
    return utility


def load_graph(path: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in rq2_read_jsonl(path):
        if (
            not bool(row["native_graph"])
            or bool(row["fallback_used"])
            or bool(row["callback_used"])
            or bool(row["padding_used"])
            or bool(row["same_source_thread"])
        ):
            raise ValueError("Graph G50 provenance or leakage invariant changed")
        grouped[str(row["query_id"])].append(
            (int(row["graph_rank"]), str(row["candidate_id"]))
        )
    graph = {
        qid: [candidate_id for _, candidate_id in sorted(rows)]
        for qid, rows in grouped.items()
    }
    if len(graph) != 300 or any(len(ids) != 50 or len(set(ids)) != 50 for ids in graph.values()):
        raise ValueError("strict G50 coverage changed")
    return graph


def load_scored_dense(path: Path, backend: str) -> dict[str, list[dict]]:
    """Load a frozen scored Dense ranking without consulting utility labels."""
    grouped: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    for row in rq2_read_jsonl(path):
        if str(row["backend"]) != backend:
            continue
        grouped[str(row["query_id"])].append(
            (int(row["rank"]), str(row["comment_id"]), float(row["score"]))
        )
    result = {
        qid: [
            {"comment_id": cid, "score": score}
            for _, cid, score in sorted(rows)
        ]
        for qid, rows in grouped.items()
    }
    if len(result) != 300 or any(
        len(rows) != 50 or len({row["comment_id"] for row in rows}) != 50
        for rows in result.values()
    ):
        raise ValueError(f"{backend}: scored D50 coverage changed")
    return result


def load_scored_graph(path: Path) -> dict[str, list[dict]]:
    """Load canonical G50 order plus a route-normalised score for CC.

    The pure-Graph and RRF arms retain the frozen two-route round-robin rank.
    CC needs a scalar Graph score, so each native route is normalised within
    query through the canonical fusion helper and the two values are averaged;
    a missing route membership contributes zero.  This remains label-blind.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rq2_read_jsonl(path):
        if (
            not bool(row["native_graph"])
            or bool(row["fallback_used"])
            or bool(row["callback_used"])
            or bool(row["padding_used"])
            or bool(row["same_source_thread"])
        ):
            raise ValueError("Graph G50 provenance or leakage invariant changed")
        grouped[str(row["query_id"])].append(row)

    result: dict[str, list[dict]] = {}
    expected_routes = ("fact_only_no_recognition", "no_recognition")
    for qid, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["graph_rank"]))
        if len(ordered) != 50 or len({str(row["candidate_id"]) for row in ordered}) != 50:
            raise ValueError(f"{qid}: scored G50 coverage changed")
        route_scores = {
            route: {
                str(row["candidate_id"]): float(row["native_graph_score"][route])
                for row in ordered
                if route in row["native_graph_score"]
            }
            for route in expected_routes
        }
        if any(not scores for scores in route_scores.values()):
            raise ValueError(f"{qid}: Graph native route scores missing")
        normalised = {
            route: normalize_scores(scores, method="minmax")
            for route, scores in route_scores.items()
        }
        result[qid] = [{
            "comment_id": str(row["candidate_id"]),
            "score": mean(
                normalised[route].get(str(row["candidate_id"]), 0.0)
                for route in expected_routes
            ),
        } for row in ordered]
    if len(result) != 300:
        raise ValueError("scored G50 query coverage changed")
    return result


def weighted_interleave(
    dense_ids: list[str],
    graph_ids: list[str],
    dense_share: float,
    target_size: int,
) -> list[str]:
    """Create one nested, exact-depth, label-blind source-ratio ranking."""
    if not 0.0 <= dense_share <= 1.0:
        raise ValueError("dense share must be in [0, 1]")
    sources = {"dense": dense_ids, "graph": graph_ids}
    indices = {"dense": 0, "graph": 0}
    counts = {"dense": 0, "graph": 0}
    selected: list[str] = []
    seen: set[str] = set()
    while len(selected) < target_size:
        next_size = len(selected) + 1
        deficit = {
            "dense": next_size * dense_share - counts["dense"],
            "graph": next_size * (1.0 - dense_share) - counts["graph"],
        }
        preferred = "dense" if deficit["dense"] >= deficit["graph"] else "graph"
        order = (preferred, "graph" if preferred == "dense" else "dense")
        added = False
        for source in order:
            rows = sources[source]
            while indices[source] < len(rows):
                candidate_id = rows[indices[source]]
                indices[source] += 1
                if candidate_id in seen:
                    continue
                selected.append(candidate_id)
                seen.add(candidate_id)
                counts[source] += 1
                added = True
                break
            if added:
                break
        if not added:
            raise ValueError("source union cannot fill the requested pool depth")
    return selected


def ordered_fusion_ids(
    dense_rows: list[dict],
    graph_rows: list[dict],
    *,
    mode: str,
    dense_weight: float,
    graph_weight: float,
    rrf_k0: int,
    cc_normalization: str,
) -> list[str]:
    runs = {"dense": dense_rows, "graph": graph_rows}
    weights = {"dense": dense_weight, "graph": graph_weight}
    if mode == "rrf":
        scores = rrf_scores(runs, weights=weights, k0=rrf_k0)
    elif mode == "cc":
        scores = cc_scores(runs, weights=weights, normalization=cc_normalization)
    else:
        raise ValueError(f"unknown fusion mode: {mode}")
    return [candidate_id for candidate_id, _ in sorted(
        scores.items(), key=lambda item: (-float(item[1]), item[0])
    )]


def pool_summary(
    *,
    pools: dict[str, list[str]],
    baseline: dict[str, list[str]],
    utility: dict[tuple[str, str], float],
    qids: list[str],
    bootstrap_indices: np.ndarray,
) -> tuple[dict, list[dict]]:
    missing = sorted(
        (qid, candidate_id)
        for qid in qids
        for candidate_id in pools[qid]
        if (qid, candidate_id) not in utility
    )
    if missing:
        raise ValueError(f"pool contains unjudged pairs: {missing[:5]} ({len(missing)} total)")
    pool_oracle = {
        qid: rq2_oracle(pools[qid], qid, utility) for qid in qids
    }
    baseline_oracle = {
        qid: rq2_oracle(baseline[qid], qid, utility) for qid in qids
    }
    deltas = [pool_oracle[qid] - baseline_oracle[qid] for qid in qids]
    delta, low, high = rq2_paired_summary(deltas, bootstrap_indices)
    pairs = [
        (qid, candidate_id)
        for qid in qids
        for candidate_id in pools[qid]
    ]
    per_query = [{
        "query_id": qid,
        "pool_size": len(pools[qid]),
        "oracle_u8": pool_oracle[qid],
        "baseline_oracle_u8": baseline_oracle[qid],
        "delta_oracle_u8": pool_oracle[qid] - baseline_oracle[qid],
    } for qid in qids]
    return {
        "N": len(qids),
        "candidate_pairs": len(pairs),
        "mean_pool_size": mean(len(pools[qid]) for qid in qids),
        "minimum_pool_size": min(len(pools[qid]) for qid in qids),
        "maximum_pool_size": max(len(pools[qid]) for qid in qids),
        "mean_pool_u": mean(utility[pair] for pair in pairs),
        "useful_count_u_ge_4": sum(utility[pair] >= 4.0 for pair in pairs),
        "useful_fraction": mean(utility[pair] >= 4.0 for pair in pairs),
        "oracle_u8": mean(pool_oracle.values()),
        "baseline_oracle_u8": mean(baseline_oracle.values()),
        "delta_oracle_u8": delta,
        "ci_low": low,
        "ci_high": high,
    }, per_query


def markdown_table(rows: list[dict], columns: list[tuple[str, str]], digits: set[str]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            values.append(f"{float(value):.4f}" if key in digits else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run(root: Path, output_dir: Path) -> dict:
    preflight = root / "out/rq2a_graph_budget_sweep_v1/preflight"
    registry_path = root / "out/rq2a_graph_budget_sweep_v1/complete/utility_registry_coverage_complete.jsonl"
    old_registry_path = root / "out/rq2a_fact_only_label_completion_v1/complete/utility_registry_coverage_complete.jsonl"
    dense_path = root / "out/development300_m50_preflight_v1/dense_m50_memberships.jsonl"
    graph_path = preflight / "graph_prefix_memberships.jsonl"
    allocation_path = preflight / "equal_total_allocation_memberships.jsonl"

    if output_dir.exists():
        raise FileExistsError(f"versioned analysis output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    old_lines = old_registry_path.read_bytes().splitlines(keepends=True)
    new_lines = registry_path.read_bytes().splitlines(keepends=True)
    if len(old_lines) != 37_581 or len(new_lines) != 44_061:
        raise ValueError("utility registry row-count invariant changed")
    if new_lines[: len(old_lines)] != old_lines:
        raise ValueError("new registry is not an exact byte-prefix append")

    utility = load_utility(registry_path)
    if len(utility) != 44_061:
        raise ValueError("utility registry unique-pair count changed")
    rankings, _ = rq2_load_rankings(dense_path)
    graph = load_graph(graph_path)
    qids = sorted(graph)
    qid_set = set(qids)
    for backend in BACKENDS:
        if set(rankings[backend]) != qid_set:
            raise ValueError(f"{backend}: Dense query coverage changed")
        if any(len(rankings[backend][qid]) != 50 for qid in qids):
            raise ValueError(f"{backend}: D50 coverage changed")

    bootstrap_indices = rq2_bootstrap_indices(
        len(qids), seed=BOOTSTRAP_SEED, dtype=np.int32
    )
    bootstrap_hash = hashlib.sha256(bootstrap_indices.tobytes()).hexdigest()
    expected_hash = "f1804695368b8bbea9d70bb9e39b4764ac22c4fdc552c40b2f7308d985f09b8a"
    if bootstrap_hash != expected_hash:
        raise ValueError("Development300 bootstrap identity changed")

    design_a: list[dict] = []
    design_b: list[dict] = []
    per_query_rows: list[dict] = []
    for backend in BACKENDS:
        dense50 = {qid: rankings[backend][qid][:50] for qid in qids}
        for dense_depth in DENSE_DEPTHS:
            dense = {qid: rankings[backend][qid][:dense_depth] for qid in qids}
            for graph_budget in DENSE_DEPTHS:
                pools = {
                    qid: stable_unique(dense[qid] + graph[qid][:graph_budget])
                    for qid in qids
                }
                stats, per_query = pool_summary(
                    pools=pools,
                    baseline=dense,
                    utility=utility,
                    qids=qids,
                    bootstrap_indices=bootstrap_indices,
                )
                added_counts = [
                    len(set(graph[qid][:graph_budget]) - set(dense[qid]))
                    for qid in qids
                ]
                row = {
                    "backend": backend,
                    "dense_depth": dense_depth,
                    "graph_budget": graph_budget,
                    "pool": f"D{dense_depth}+G{graph_budget}",
                    **stats,
                    "mean_unique_graph_additions": mean(added_counts),
                    "mean_unique_graph_share": mean(
                        added / len(pools[qid])
                        for qid, added in zip(qids, added_counts)
                    ),
                }
                row["oracle_marginal_per_unique_graph_candidate"] = (
                    row["delta_oracle_u8"] / row["mean_unique_graph_additions"]
                )
                design_a.append(row)
                per_query_rows.extend({
                    "design": "A",
                    "backend": backend,
                    "dense_depth": dense_depth,
                    "graph_budget": graph_budget,
                    **item,
                } for item in per_query)

        for graph_budget in GRAPH_BUDGETS:
            pools = {
                qid: stable_unique(dense50[qid] + graph[qid][:graph_budget])
                for qid in qids
            }
            stats, per_query = pool_summary(
                pools=pools,
                baseline=dense50,
                utility=utility,
                qids=qids,
                bootstrap_indices=bootstrap_indices,
            )
            added_counts = [
                len(set(graph[qid][:graph_budget]) - set(dense50[qid]))
                for qid in qids
            ]
            row = {
                "backend": backend,
                "dense_depth": 50,
                "graph_budget": graph_budget,
                "pool": f"D50+G{graph_budget}",
                **stats,
                "mean_unique_graph_additions": mean(added_counts),
                "mean_unique_graph_share": mean(
                    added / len(pools[qid])
                    for qid, added in zip(qids, added_counts)
                ),
            }
            row["oracle_marginal_per_unique_graph_candidate"] = (
                row["delta_oracle_u8"] / row["mean_unique_graph_additions"]
            )
            design_b.append(row)
            per_query_rows.extend({
                "design": "B",
                "backend": backend,
                "dense_depth": 50,
                "graph_budget": graph_budget,
                **item,
            } for item in per_query)

    allocation_grouped: dict[tuple[str, int, int], dict[str, list[tuple[int, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rq2_read_jsonl(allocation_path):
        key = (
            str(row["backend"]),
            int(row["planned_dense_quota"]),
            int(row["planned_graph_quota"]),
        )
        allocation_grouped[key][str(row["query_id"])].append(
            (int(row["pool_rank"]), str(row["candidate_id"]))
        )
    allocation_pools = {
        key: {
            qid: [candidate_id for _, candidate_id in sorted(rows)]
            for qid, rows in by_query.items()
        }
        for key, by_query in allocation_grouped.items()
    }
    if len(allocation_pools) != 12:
        raise ValueError("fixed-total allocation grid changed")

    design_c: list[dict] = []
    for backend in BACKENDS:
        baseline = allocation_pools[(backend, 50, 0)]
        for key in sorted(
            (key for key in allocation_pools if key[0] == backend),
            key=lambda item: -item[1],
        ):
            _, dense_quota, graph_quota = key
            pools = allocation_pools[key]
            if set(pools) != qid_set or any(len(ids) != 50 for ids in pools.values()):
                raise ValueError(f"fixed-total pool invariant changed: {key}")
            stats, per_query = pool_summary(
                pools=pools,
                baseline=baseline,
                utility=utility,
                qids=qids,
                bootstrap_indices=bootstrap_indices,
            )
            graph_contributions = [
                len(set(graph[qid][:graph_quota]) - set(rankings[backend][qid][:dense_quota]))
                for qid in qids
            ]
            design_c.append({
                "backend": backend,
                "planned_dense_quota": dense_quota,
                "planned_graph_quota": graph_quota,
                "allocation": f"{dense_quota}+{graph_quota}",
                **stats,
                "mean_unique_graph_contribution": mean(graph_contributions),
            })
            per_query_rows.extend({
                "design": "C",
                "backend": backend,
                "planned_dense_quota": dense_quota,
                "planned_graph_quota": graph_quota,
                **item,
            } for item in per_query)

    write_csv_rows(output_dir / "design_a_symmetric_depth_grid.csv", design_a)
    write_csv_rows(output_dir / "design_b_dense50_graph_budget_sweep.csv", design_b)
    write_csv_rows(output_dir / "design_c_equal_total_allocation.csv", design_c)
    with (output_dir / "per_query_oracle_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_query_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = [
        "# RQ2a Graph-budget sweep results",
        "",
        "All results use Development300, the unchanged utility-v2 labels, K=8, "
        "and the frozen label-blind candidate memberships. CIs are 5,000-draw "
        "whole-query paired bootstrap intervals.",
        "",
        "## Design B: fixed D50, increasing Graph budget",
        "",
        markdown_table(
            design_b,
            [
                ("backend", "Backend"), ("pool", "Pool"),
                ("mean_unique_graph_additions", "Unique G"),
                ("oracle_u8", "Oracle U@8"),
                ("delta_oracle_u8", "Delta vs D50"),
                ("ci_low", "CI low"), ("ci_high", "CI high"),
            ],
            {"mean_unique_graph_additions", "oracle_u8", "delta_oracle_u8", "ci_low", "ci_high"},
        ),
        "",
        "## Design C: fixed total pool size 50",
        "",
        markdown_table(
            design_c,
            [
                ("backend", "Backend"), ("allocation", "Dense+Graph"),
                ("mean_unique_graph_contribution", "Unique G"),
                ("oracle_u8", "Oracle U@8"),
                ("delta_oracle_u8", "Delta vs 50+0"),
                ("ci_low", "CI low"), ("ci_high", "CI high"),
            ],
            {"mean_unique_graph_contribution", "oracle_u8", "delta_oracle_u8", "ci_low", "ci_high"},
        ),
        "",
        "Design A is preserved in `design_a_symmetric_depth_grid.csv` because the full 32-row grid is more readable as data than as Markdown.",
        "",
    ]
    report_path = output_dir / "RESULTS.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    output_names = (
        "design_a_symmetric_depth_grid.csv",
        "design_b_dense50_graph_budget_sweep.csv",
        "design_c_equal_total_allocation.csv",
        "per_query_oracle_results.jsonl",
        "RESULTS.md",
    )
    manifest = {
        "schema": "rq2a-graph-budget-sweep-analysis-v1",
        "status": "COMPLETE",
        "counts": {
            "development_queries": len(qids),
            "utility_pairs": len(utility),
            "design_a_rows": len(design_a),
            "design_b_rows": len(design_b),
            "design_c_rows": len(design_c),
            "per_query_rows": len(per_query_rows),
        },
        "estimand": "candidate access under Oracle U@8",
        "statistics": {
            "final_evidence_budget_k": FINAL_K,
            "bootstrap_unit": "whole query",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_indices_sha256": bootstrap_hash,
        },
        "boundaries": {
            "external_requests_made": 0,
            "frozen_test_read": False,
            "candidate_construction_used_utility": False,
            "realised_selector_evaluated": False,
        },
        "inputs": {
            "utility_registry": {"path": str(registry_path), "sha256": rq2_sha256(registry_path)},
            "dense_memberships": {"path": str(dense_path), "sha256": rq2_sha256(dense_path)},
            "graph_memberships": {"path": str(graph_path), "sha256": rq2_sha256(graph_path)},
            "allocation_memberships": {"path": str(allocation_path), "sha256": rq2_sha256(allocation_path)},
        },
        "outputs": {
            name: rq2_sha256(output_dir / name) for name in output_names
        },
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_stage1_figure(
    root: Path,
    output_dir: Path,
    config_key: str = "rq2a_stage1_candidate_pool_figure",
) -> dict:
    """Build and render the requested nested candidate-pool curves.

    The base seven methods are fixed.  A configuration may add extra,
    externally constructed rankings (``extra_methods``: key, label, rankings
    jsonl of {query_id, ordered_ids}); their oracle is computed over labelled
    candidates only (a lower bound when label coverage is incomplete), so an
    extra curve never widens the label-blind base contract.
    """
    cfg = params(config_key)
    if not isinstance(cfg, dict):
        raise ValueError(f"{config_key} config is missing")
    if bool(cfg["allow_external_calls"]) or bool(cfg["allow_frozen_test"]):
        raise ValueError("Stage-1 figure must remain local and development-only")
    if output_dir.exists():
        raise FileExistsError(f"versioned Stage-1 output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    backend = str(cfg["backend"])
    minimum_depth = int(cfg["minimum_pool_depth"])
    maximum_depth = int(cfg["maximum_pool_depth"])
    final_k = int(params("retrieval", "k"))
    if minimum_depth < final_k or maximum_depth != 50:
        raise ValueError("Stage-1 figure requires K<=M and the judged D50/G50 boundary")
    depths = list(range(minimum_depth, maximum_depth + 1))
    displayed_depths = [int(value) for value in cfg["displayed_depths"]]
    if any(value not in depths for value in displayed_depths):
        raise ValueError("displayed depths must lie on the evaluated depth grid")

    preflight_cfg = params(str(cfg.get(
        "preflight_config_key", "rq2a_graph_budget_sweep_preflight"
    )))
    realised_cfg = params(str(cfg.get(
        "realised_selector_config_key", "rq2a_graph_budget_realised_selector"
    )))
    dense_path = root / str(cfg.get(
        "dense_memberships",
        "out/development300_m50_preflight_v1/dense_m50_memberships.jsonl",
    ))
    graph_path = root / str(cfg.get(
        "graph_memberships", realised_cfg["graph_memberships"]
    ))
    utility_path = root / str(cfg.get(
        "utility_registry", realised_cfg["utility_registry"]
    ))

    # Construct every method ranking before utility is loaded.  This ordering
    # makes the label-blind boundary executable rather than merely declarative.
    dense = load_scored_dense(dense_path, backend)
    graph = load_scored_graph(graph_path)
    qids = sorted(graph)
    if set(dense) != set(graph):
        raise ValueError("Dense/Graph query coverage mismatch")

    ratios = cfg["dense_graph_ratios"]
    fusion_cfg = cfg["fusion"]
    canonical_fusion_weights = params("fusion", "weights")
    dense_weight = float(canonical_fusion_weights["semantic"])
    graph_weight = float(canonical_fusion_weights["multihop"])
    rrf_k0 = int(params("retrieval", "k0"))
    method_rankings: dict[str, dict[str, list[str]]] = {
        key: {} for key, _ in STAGE1_METHODS
    }
    for qid in qids:
        dense_rows = dense[qid]
        graph_rows = graph[qid]
        dense_ids = [str(row["comment_id"]) for row in dense_rows]
        graph_ids = [str(row["comment_id"]) for row in graph_rows]
        method_rankings["dense_only"][qid] = dense_ids
        method_rankings["graph_only"][qid] = graph_ids
        for method in (
            "dense_70_graph_30",
            "dense_50_graph_50",
            "dense_30_graph_70",
        ):
            dense_share, graph_share = (float(value) for value in ratios[method])
            if not math.isclose(dense_share + graph_share, 1.0):
                raise ValueError(f"{method}: source shares must sum to one")
            method_rankings[method][qid] = weighted_interleave(
                dense_ids, graph_ids, dense_share, maximum_depth
            )
        for mode in ("rrf", "cc"):
            method_rankings[mode][qid] = ordered_fusion_ids(
                dense_rows,
                graph_rows,
                mode=mode,
                dense_weight=dense_weight,
                graph_weight=graph_weight,
                rrf_k0=rrf_k0,
                cc_normalization=str(fusion_cfg["cc_normalization"]),
            )[:maximum_depth]

    extras: list[tuple[str, str]] = []
    extra_inputs: dict[str, dict] = {}
    extra_style: dict[str, dict] = {}
    for spec in list(cfg.get("extra_methods") or []):
        key, label = str(spec["key"]), str(spec["label"])
        rankings_path = root / str(spec["rankings"])
        per_query: dict[str, list[str]] = {}
        with rankings_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                per_query[str(row["query_id"])] = [
                    str(cid) for cid in row["ordered_ids"]
                ]
        method_rankings[key] = per_query
        extras.append((key, label))
        extra_inputs[key] = {
            "path": str(rankings_path), "sha256": rq2_sha256(rankings_path),
        }
        extra_style[key] = {
            "color": str(spec.get("color", "#B22222")),
            "marker": str(spec.get("marker", "*")),
        }
    all_methods = list(STAGE1_METHODS) + extras
    extra_keys = {key for key, _ in extras}

    for method, _ in all_methods:
        rankings = method_rankings[method]
        if set(rankings) != set(qids) or any(
            len(ids) != maximum_depth or len(set(ids)) != maximum_depth
            for ids in rankings.values()
        ):
            raise ValueError(f"{method}: exact nested M50 ranking invariant changed")

    utility = load_utility(utility_path)
    expected_utility_pairs = int(cfg.get("expected_utility_pairs", 44_061))
    if len(utility) != expected_utility_pairs:
        raise ValueError("coverage-complete utility registry changed")
    missing = sorted(
        (qid, candidate_id)
        for method, _ in STAGE1_METHODS
        for qid in qids
        for candidate_id in method_rankings[method][qid]
        if (qid, candidate_id) not in utility
    )
    if missing:
        raise ValueError(f"Stage-1 ranking contains unjudged pairs: {missing[:5]}")

    bootstrap_seed = int(cfg["bootstrap_seed"])
    bootstrap_samples = int(cfg["bootstrap_samples"])
    bootstrap_indices = rq2_bootstrap_indices(
        len(qids), seed=bootstrap_seed, dtype=np.int32
    )
    if bootstrap_indices.shape != (bootstrap_samples, len(qids)):
        raise ValueError("Stage-1 bootstrap plan changed")
    bootstrap_hash = hashlib.sha256(bootstrap_indices.tobytes()).hexdigest()

    def labelled_only_oracle(ids: list[str], qid: str) -> float:
        values = sorted(
            (utility[(qid, cid)] for cid in ids if (qid, cid) in utility),
            reverse=True,
        )[:final_k]
        return sum(values) / final_k

    def method_oracle(method: str, ids: list[str], qid: str) -> float:
        if method in extra_keys:
            return labelled_only_oracle(ids, qid)
        return rq2_oracle(ids, qid, utility)

    extra_coverage = {
        key: {
            "ranked_pairs": sum(
                len(method_rankings[key][qid]) for qid in qids
            ),
            "unlabelled_pairs": sum(
                1 for qid in qids for cid in method_rankings[key][qid]
                if (qid, cid) not in utility
            ),
        }
        for key, _ in extras
    }

    curve_rows: list[dict] = []
    dense_oracle_by_depth: dict[int, np.ndarray] = {}
    for method, label in all_methods:
        previous = -math.inf
        for depth in depths:
            oracle_values = np.asarray([
                method_oracle(method, method_rankings[method][qid][:depth], qid)
                for qid in qids
            ], dtype=float)
            value = float(oracle_values.mean())
            if value + 1e-12 < previous:
                raise AssertionError(f"{method}: Oracle curve is not monotone")
            previous = value
            if method == "dense_only":
                dense_oracle_by_depth[depth] = oracle_values
            dense_baseline = dense_oracle_by_depth.get(depth)
            if dense_baseline is None:
                raise AssertionError("Dense baseline must be evaluated first")
            delta, delta_low, delta_high = rq2_paired_summary(
                list(oracle_values - dense_baseline), bootstrap_indices
            )
            bootstrap_means = oracle_values[bootstrap_indices].mean(axis=1)
            pool_values = [
                utility[(qid, candidate_id)]
                for qid in qids
                for candidate_id in method_rankings[method][qid][:depth]
                if (qid, candidate_id) in utility
            ]
            curve_rows.append({
                "backend": backend,
                "method": method,
                "method_label": label,
                "pool_depth": depth,
                "oracle_u8": value,
                "oracle_ci_low": float(np.quantile(bootstrap_means, 0.025)),
                "oracle_ci_high": float(np.quantile(bootstrap_means, 0.975)),
                "delta_vs_dense": delta,
                "delta_vs_dense_ci_low": delta_low,
                "delta_vs_dense_ci_high": delta_high,
                "mean_candidate_utility": mean(pool_values),
                "query_count": len(qids),
                "pool_pairs": len(pool_values),
                "final_evidence_budget_k": final_k,
            })

    write_csv_rows(output_dir / "stage1_candidate_pool_oracle_curves.csv", curve_rows)
    final_rows = [row for row in curve_rows if int(row["pool_depth"]) == maximum_depth]
    write_csv_rows(output_dir / "stage1_candidate_pool_oracle_at_m50.csv", final_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    colours = {
        "dense_only": "#0072B2",
        "dense_70_graph_30": "#009E73",
        "dense_50_graph_50": "#E69F00",
        "dense_30_graph_70": "#D55E00",
        "graph_only": "#CC79A7",
        "rrf": "#56B4E9",
        "cc": "#6F4C9B",
    }
    markers = {
        "dense_only": "o",
        "dense_70_graph_30": "s",
        "dense_50_graph_50": "D",
        "dense_30_graph_70": "^",
        "graph_only": "v",
        "rrf": "P",
        "cc": "X",
    }
    for key, _ in extras:
        colours[key] = extra_style[key]["color"]
        markers[key] = extra_style[key]["marker"]
    presentation = cfg["presentation"]
    figure_size = tuple(float(value) for value in presentation["figure_size"])
    main_line_width = float(presentation["main_line_width"])
    marker_size = float(presentation["marker_size"])
    delta_panel_methods = [str(value) for value in presentation["delta_panel_methods"]]
    if delta_panel_methods != ["rrf", "cc"]:
        raise ValueError("Stage-1 detail panel must compare RRF and CC with Dense")

    mark_indices = [depths.index(value) for value in displayed_depths]
    fig, (ax, delta_ax) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=figure_size,
        gridspec_kw={"height_ratios": [3.4, 1.15], "hspace": 0.16},
    )
    handles = []
    for method, label in all_methods:
        rows = [row for row in curve_rows if row["method"] == method]
        handle, = ax.plot(
            [int(row["pool_depth"]) for row in rows],
            [float(row["oracle_u8"]) for row in rows],
            color=colours[method],
            linewidth=main_line_width,
            marker=markers[method],
            markersize=marker_size,
            markeredgecolor="white",
            markeredgewidth=0.55,
            markevery=mark_indices,
            label=label,
            zorder=3,
        )
        handles.append(handle)

    y_values = [float(row["oracle_u8"]) for row in curve_rows]
    y_low = math.floor((min(y_values) - 0.04) * 10.0) / 10.0
    y_high = math.ceil((max(y_values) + 0.04) * 10.0) / 10.0
    ax.set_xlim(minimum_depth - 0.5, maximum_depth + 0.5)
    ax.set_ylim(y_low, y_high)
    ax.set_xticks(displayed_depths)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_ylabel("Oracle Utility@8", fontsize=11.5, labelpad=8)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.85, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(colors="#333333", labelsize=10.5)

    fusion_delta_values: list[float] = []
    for method in delta_panel_methods:
        rows = [row for row in curve_rows if row["method"] == method]
        values = [float(row["delta_vs_dense"]) for row in rows]
        fusion_delta_values.extend(values)
        delta_ax.plot(
            [int(row["pool_depth"]) for row in rows],
            values,
            color=colours[method],
            linewidth=1.35,
            marker=markers[method],
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markevery=mark_indices,
            zorder=3,
        )
    delta_low = math.floor((min(fusion_delta_values) - 0.004) * 100.0) / 100.0
    delta_high = math.ceil((max(fusion_delta_values) + 0.004) * 100.0) / 100.0
    delta_ax.set_ylim(delta_low, delta_high)
    delta_ax.yaxis.set_major_locator(MultipleLocator(0.01))
    delta_ax.axhline(0.0, color="#555555", linewidth=0.85, zorder=1)
    delta_ax.set_xlabel("Candidate-pool depth, M", fontsize=11.5, labelpad=8)
    delta_ax.set_ylabel("Δ vs Dense", fontsize=10.5, labelpad=9)
    delta_ax.grid(axis="y", color="#E0E0E0", linewidth=0.7, alpha=0.9, zorder=0)
    delta_ax.grid(axis="x", visible=False)
    delta_ax.spines["top"].set_visible(False)
    delta_ax.spines["right"].set_visible(False)
    delta_ax.spines["left"].set_color("#555555")
    delta_ax.spines["bottom"].set_color("#555555")
    delta_ax.tick_params(colors="#333333", labelsize=9.5)
    delta_ax.annotate(
        "RRF = Dense",
        xy=(maximum_depth, 0.0),
        xytext=(42.5, delta_high - 0.008),
        color=colours["rrf"],
        fontsize=8.8,
        ha="left",
        arrowprops={
            "arrowstyle": "->",
            "color": colours["rrf"],
            "linewidth": 0.8,
            "shrinkA": 2,
            "shrinkB": 3,
        },
    )

    fig.suptitle(
        "Stage 1 — Candidate-pool access",
        x=0.10,
        y=0.985,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color="#222222",
    )
    fig.text(
        0.10,
        0.936,
        str(cfg.get(
            "subtitle",
            "Development300 · E5 used for the Dense arm · higher is better",
        )),
        ha="left",
        fontsize=10.5,
        color="#555555",
    )
    legend_order = (0, 4, 1, 5, 2, 6, 3) + tuple(
        range(len(STAGE1_METHODS), len(all_methods))
    )
    fig.legend(
        handles=[handles[index] for index in legend_order],
        labels=[all_methods[index][1] for index in legend_order],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.902),
        ncol=4,
        frameon=False,
        fontsize=9.4,
        handlelength=2.5,
        columnspacing=1.5,
        handletextpad=0.6,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.11, top=0.745)

    png_path = output_dir / "stage1_candidate_pool_oracle_curves_e5.png"
    pdf_path = output_dir / "stage1_candidate_pool_oracle_curves_e5.pdf"
    fig.savefig(png_path, dpi=320, facecolor="white")
    fig.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Title": "Stage 1 — Candidate-pool access",
            "Creator": "GraphRAG ADHD project",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)

    caption = (
        "**Figure. Stage 1 candidate-pool access on Development300.** "
        "For each pool depth M, Oracle Utility@8 is the mean utility of the "
        "best eight candidates available in the pool. Dense:Graph ratio lines "
        "are deterministic nested weighted interleavings. RRF and CC fuse the "
        f"frozen Dense and Graph rankings with weights "
        f"{dense_weight:g}:{graph_weight:g} "
        f"(RRF k0={rrf_k0}; CC uses query-local "
        f"{fusion_cfg['cc_normalization']} normalisation). Candidate construction "
        "is label-blind; utility is used only after every ranking is frozen. "
        "The lower panel magnifies the paired point-estimate difference between "
        "each fusion method and Dense at the same M; the annotation marks the "
        "M=50 point where RRF and Dense expose the same candidate set. Higher is "
        "better. Confidence intervals are retained in the "
        "companion CSV but omitted from the figure to keep seven method curves legible.\n"
    )
    caption_note = str(cfg.get("caption_note") or "").strip()
    if caption_note:
        caption += "\n" + caption_note + "\n"
    (output_dir / "CAPTION.md").write_text(caption, encoding="utf-8")

    output_names = (
        "stage1_candidate_pool_oracle_curves.csv",
        "stage1_candidate_pool_oracle_at_m50.csv",
        "stage1_candidate_pool_oracle_curves_e5.png",
        "stage1_candidate_pool_oracle_curves_e5.pdf",
        "CAPTION.md",
    )
    curve_lookup = {
        (str(row["method"]), int(row["pool_depth"])): row for row in curve_rows
    }
    selected_results = {
        "rrf_vs_dense_m20": {
            "delta": curve_lookup[("rrf", 20)]["delta_vs_dense"],
            "ci_low": curve_lookup[("rrf", 20)]["delta_vs_dense_ci_low"],
            "ci_high": curve_lookup[("rrf", 20)]["delta_vs_dense_ci_high"],
        },
        "rrf_vs_dense_m30": {
            "delta": curve_lookup[("rrf", 30)]["delta_vs_dense"],
            "ci_low": curve_lookup[("rrf", 30)]["delta_vs_dense_ci_low"],
            "ci_high": curve_lookup[("rrf", 30)]["delta_vs_dense_ci_high"],
        },
        "dense70_graph30_vs_dense_m50": {
            "delta": curve_lookup[("dense_70_graph_30", 50)]["delta_vs_dense"],
            "ci_low": curve_lookup[("dense_70_graph_30", 50)]["delta_vs_dense_ci_low"],
            "ci_high": curve_lookup[("dense_70_graph_30", 50)]["delta_vs_dense_ci_high"],
        },
        "rrf_equals_dense_m50": math.isclose(
            curve_lookup[("rrf", 50)]["oracle_u8"],
            curve_lookup[("dense_only", 50)]["oracle_u8"],
        ),
    }
    manifest = {
        "schema": "rq2a-stage1-candidate-pool-curves-v1",
        "version": str(cfg["version"]),
        "status": "COMPLETE",
        "counts": {
            "development_queries": len(qids),
            "methods": len(all_methods),
            "depths": len(depths),
            "curve_rows": len(curve_rows),
            "utility_pairs": len(utility),
        },
        "estimand": "candidate access under Oracle U@8",
        "method_order": [method for method, _ in all_methods],
        **({"extra_methods": {
            key: {
                "label": label,
                "oracle": "labelled candidates only (lower bound if coverage < 100%)",
                **extra_coverage[key],
            } for key, label in extras
        }} if extras else {}),
        "config": {
            "backend": backend,
            "depth_range": [minimum_depth, maximum_depth],
            "displayed_depths": displayed_depths,
            "dense_graph_ratios": ratios,
            "fusion": {
                "weights_source": "configuration/params.yaml::fusion.weights",
                "dense_weight": dense_weight,
                "graph_weight": graph_weight,
                "rrf_k0_source": "configuration/params.yaml::retrieval.k0",
                "rrf_k0": rrf_k0,
                "cc_normalization": str(fusion_cfg["cc_normalization"]),
            },
            "final_evidence_budget_k": final_k,
            "expected_utility_pairs": expected_utility_pairs,
            "presentation": {
                "figure_size": list(figure_size),
                "main_line_width": main_line_width,
                "marker_size": marker_size,
                "delta_panel_estimand": "fusion Oracle U@8 minus Dense Oracle U@8 at matched M",
                "delta_panel_methods": delta_panel_methods,
            },
        },
        "statistics": {
            "bootstrap_unit": "whole query",
            "bootstrap_draws": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_indices_sha256": bootstrap_hash,
            "interval": "percentile 95% CI for the absolute query mean",
        },
        "selected_results": selected_results,
        "checks": {
            "all_methods_exact_m50_unique": True,
            "all_depths_are_nested_prefixes": True,
            "all_curve_rows_monotone": True,
            "all_ranked_pairs_utility_complete": True,
        },
        "boundaries": {
            "external_requests_made": 0,
            "frozen_test_read": False,
            "candidate_construction_used_utility": False,
            "utility_used_only_after_rankings_frozen": True,
            "realised_selector_evaluated": False,
        },
        "inputs": {
            "dense_memberships": {"path": str(dense_path), "sha256": rq2_sha256(dense_path)},
            "graph_memberships": {"path": str(graph_path), "sha256": rq2_sha256(graph_path)},
            "utility_registry": {"path": str(utility_path), "sha256": rq2_sha256(utility_path)},
            "graph_definition": str(preflight_cfg["version"]),
            **({f"extra_rankings_{key}": value
                for key, value in extra_inputs.items()} if extras else {}),
        },
        "software": {"matplotlib": matplotlib.__version__},
        "outputs": {name: rq2_sha256(output_dir / name) for name in output_names},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--stage1-figure",
        action="store_true",
        help="render the thesis-facing seven-method Stage-1 candidate-pool figure",
    )
    parser.add_argument(
        "--figure-config-key",
        default="rq2a_stage1_candidate_pool_figure",
        help="config block for the Stage-1 figure (supports extra_methods)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage1_figure:
        default_output = root / str(
            params(args.figure_config_key, "output_dir")
        )
        output_dir = (args.output_dir or default_output).resolve()
        manifest = run_stage1_figure(root, output_dir, args.figure_config_key)
    else:
        default_output = root / "out/rq2a_graph_budget_sweep_v1/analysis"
        output_dir = (args.output_dir or default_output).resolve()
        manifest = run(root, output_dir)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
