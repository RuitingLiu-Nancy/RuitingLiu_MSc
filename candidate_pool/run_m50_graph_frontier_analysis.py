#!/usr/bin/env python3
"""Fixed/residual Graph4 marginal analysis on the frozen M<=50 Dense arms."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

try:
    from evaluation.judgment_completeness import complete_utility_v2_rows
    from candidate_pool.run_dense_semantic_drift_rescue_audit import (
        _build_idf,
        _load_embeddings,
    )
    from candidate_pool.run_m50_dense_frontier_analysis import (
        BACKENDS,
        DEPTHS,
        TOP_K,
        aggregate_metrics,
        run_selector_arm,
        static_features_for_arm,
        write_csv,
    )
    from utility_scoring.annotation.run_top3_residual_judging import (
        read_jsonl,
        sha256,
        utc_now,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.judgment_completeness import complete_utility_v2_rows
    from candidate_pool.run_dense_semantic_drift_rescue_audit import (
        _build_idf,
        _load_embeddings,
    )
    from candidate_pool.run_m50_dense_frontier_analysis import (
        BACKENDS,
        DEPTHS,
        TOP_K,
        aggregate_metrics,
        run_selector_arm,
        static_features_for_arm,
        write_csv,
    )
    from utility_scoring.annotation.run_top3_residual_judging import (
        read_jsonl,
        sha256,
        utc_now,
        write_json,
        write_jsonl,
    )


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean_oof(rows: list[dict], field: str) -> float:
    by_query: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_id"])].append(float(row[field]))
    return statistics.fmean(
        statistics.fmean(values) for values in by_query.values()
    )


def oracle_at8(
    query_id: str,
    ids: list[str],
    registry: dict[tuple[str, str], dict],
) -> float:
    ordered = sorted(
        dict.fromkeys(ids),
        key=lambda cid: (float(registry[(query_id, cid)]["utility"]), cid),
        reverse=True,
    )[:TOP_K]
    if len(ordered) != TOP_K:
        raise ValueError(f"{query_id}: oracle pool has fewer than 8 candidates")
    return statistics.fmean(
        float(registry[(query_id, cid)]["utility"]) for cid in ordered
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "out/depth_graph_utility_community_frontier_dev100_m50_graph_v1",
    )
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)

    preflight = root / "out/depth_graph_utility_community_frontier_dev100_m50_v2"
    dense_dir = root / "out/depth_graph_utility_community_frontier_dev100_m50_dense_v1"
    registry_path = (
        root
        / "out/depth_graph_utility_community_frontier_dev100_m50_judging_v1"
        / "complete/utility_registry_coverage_complete.jsonl"
    )
    complete, registry = complete_utility_v2_rows(read_jsonl(registry_path))
    if len(complete) != 19813:
        raise ValueError("coverage-complete registry identity changed")

    membership = read_jsonl(
        preflight / "m50_residual_preflight/dense_m50_memberships.jsonl"
    )
    rankings: dict[str, dict[str, list[dict]]] = {
        backend: defaultdict(list) for backend in BACKENDS
    }
    for row in membership:
        rankings[str(row["backend"])][str(row["query_id"])].append(row)
    for backend in BACKENDS:
        rankings[backend] = dict(rankings[backend])
        for query_id in rankings[backend]:
            rankings[backend][query_id].sort(key=lambda row: int(row["rank"]))

    graph_rows = read_jsonl(
        preflight / "m50_residual_preflight/graph_candidate_views.jsonl"
    )
    fixed: dict[str, list[str]] = defaultdict(list)
    residual: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for row in graph_rows:
        query_id = str(row["query_id"])
        candidate_id = str(row["comment_id"])
        if not (
            row["native_graph"]
            and not row["fallback_used"]
            and not row["callback_used"]
            and not row["padding_used"]
        ):
            raise ValueError("Graph provenance changed")
        if row["view_type"] == "fixed_graph4":
            fixed[query_id].append(candidate_id)
        else:
            residual[(str(row["backend"]), int(row["depth"]), query_id)].append(
                candidate_id
            )

    split_path = (
        root / "out/strict_native_graph_conservative_policy_dev100_v1/split_manifest.json"
    )
    splits = json.loads(split_path.read_text(encoding="utf-8"))["rows"]
    query_ids = sorted(fixed)
    (
        candidate_vectors,
        query_vectors,
        corpus_text,
        query_text,
        _,
        _,
    ) = _load_embeddings(
        corpus_path=root / "out/hipporag_official_adapter/adhd_peer_support_validation_corpus.json",
        corpus_embeddings_path=root / "out/sbert_graph_comparison_dev100_v1/sbert_corpus_embeddings.npy",
        queries_path=root / "out/section17_dev_100_v2/section17_dev_queries_official.json",
        query_embeddings_path=root / "out/sbert_graph_comparison_dev100_v1/sbert_query_embeddings.npy",
        qids=set(query_ids),
    )
    idf, _ = _build_idf(corpus_text.values())

    dense_metrics = {
        (row["backend"], int(row["depth"])): row
        for row in read_csv(dense_dir / "m50_depth_metrics.csv")
    }
    absorption_rows, residual_rows, matched_rows = [], [], []
    action_rows, all_oof, fold_audit = [], [], []
    for backend in BACKENDS:
        baseline_ids = {
            qid: [
                str(row["comment_id"]) for row in rankings[backend][qid][:TOP_K]
            ]
            for qid in query_ids
        }
        for depth in DEPTHS:
            dense_m = {
                qid: [
                    str(row["comment_id"])
                    for row in rankings[backend][qid][:depth]
                ]
                for qid in query_ids
            }
            dense_tail = {
                qid: dense_m[qid][TOP_K:] for qid in query_ids
            }
            graph_views = {
                "fixed_graph4": {qid: fixed[qid] for qid in query_ids},
                "residual_graph4": {
                    qid: residual[(backend, depth, qid)] for qid in query_ids
                },
            }
            for view_name, graph_ids in graph_views.items():
                union_candidates = {}
                for qid in query_ids:
                    baseline_set = set(baseline_ids[qid])
                    union_candidates[qid] = [
                        cid
                        for cid in dict.fromkeys([
                            *dense_tail[qid],
                            *graph_ids[qid],
                        ])
                        if cid not in baseline_set
                    ]
                rank_maps = {
                    qid: {
                        str(row["comment_id"]): int(row["rank"])
                        for row in rankings[backend][qid][:depth]
                    }
                    for qid in query_ids
                }
                arm_ids = {
                    qid: [*baseline_ids[qid], *union_candidates[qid]]
                    for qid in query_ids
                }
                static = static_features_for_arm(
                    query_ids=query_ids,
                    baseline_ids=baseline_ids,
                    arm_ids=arm_ids,
                    rank_maps=rank_maps,
                    candidate_vectors=candidate_vectors,
                    query_vectors=query_vectors,
                    corpus_text=corpus_text,
                    query_text=query_text,
                    idf=idf,
                )
                oof, audit = run_selector_arm(
                    backend=backend,
                    depth=depth,
                    query_ids=query_ids,
                    split_rows=splits,
                    baseline_ids=baseline_ids,
                    candidate_ids=union_candidates,
                    static=static,
                    candidate_vectors=candidate_vectors,
                    registry=registry,
                    pool_name=f"{backend}_D{depth}_plus_{view_name}",
                )
                all_oof.extend(oof)
                fold_audit.extend(audit)
                dense_realised = float(
                    dense_metrics[(backend, depth)]["realised_oof_utility_at8"]
                )
                union_realised = mean_oof(oof, "policy_utility_at8")
                graph_sets = {qid: set(graph_ids[qid]) for qid in query_ids}
                graph_selected = [
                    row for row in oof
                    if row.get("selected_candidate_id") in graph_sets[
                        str(row["query_id"])
                    ]
                ]
                graph_action_mass = sum(float(row["acted"]) for row in graph_selected)
                graph_success_mass = sum(
                    float(row["successful_action"]) for row in graph_selected
                )
                graph_harm_mass = sum(
                    float(row["harmful_action"]) for row in graph_selected
                )
                dense_oracle_values, union_oracle_values = [], []
                overlaps, graph_unique = [], []
                matched_oracle_values = []
                for qid in query_ids:
                    dense_set = set(dense_m[qid])
                    overlaps.append(len(dense_set & graph_sets[qid]))
                    graph_unique.append(len(graph_sets[qid] - dense_set))
                    dense_oracle_values.append(
                        oracle_at8(qid, dense_m[qid], registry)
                    )
                    union_oracle_values.append(
                        oracle_at8(
                            qid,
                            [*dense_m[qid], *graph_ids[qid]],
                            registry,
                        )
                    )
                    matched_dense_count = max(4, depth - 4)
                    matched_ids = list(
                        dict.fromkeys([
                            *dense_m[qid][:matched_dense_count],
                            *graph_ids[qid],
                        ])
                    )
                    # Graph/Dense overlap must not silently shrink the matched
                    # candidate budget.  Fill only from the already in-scope
                    # Dense Top-M order until the total unique budget is M.
                    for candidate_id in dense_m[qid]:
                        if len(matched_ids) >= depth:
                            break
                        if candidate_id not in matched_ids:
                            matched_ids.append(candidate_id)
                    matched_oracle_values.append(
                        oracle_at8(
                            qid,
                            matched_ids,
                            registry,
                        )
                    )
                common = {
                    "backend": backend,
                    "depth": depth,
                    "graph_view": view_name,
                    "graph_rows": 400,
                    "mean_dense_overlap_count": statistics.fmean(overlaps),
                    "mean_graph_unique_count": statistics.fmean(graph_unique),
                    "graph_unique_fraction": statistics.fmean(graph_unique) / 4.0,
                    "dense_oracle_utility_at8": statistics.fmean(
                        dense_oracle_values
                    ),
                    "union_oracle_utility_at8": statistics.fmean(
                        union_oracle_values
                    ),
                    "oracle_marginal_vs_dense": statistics.fmean(
                        union_oracle_values
                    ) - statistics.fmean(dense_oracle_values),
                    "dense_realised_utility_at8": dense_realised,
                    "union_realised_utility_at8": union_realised,
                    "realised_marginal_vs_dense": union_realised - dense_realised,
                    "graph_selected_action_mass": graph_action_mass / 5.0,
                    "graph_entrant_precision": (
                        graph_success_mass / graph_action_mass
                        if graph_action_mass else None
                    ),
                    "graph_harmful_action_rate_all_queries": (
                        graph_harm_mass / (len(query_ids) * 5)
                    ),
                    "matched_budget_oracle_utility_at8": statistics.fmean(
                        matched_oracle_values
                    ),
                    "fallback_count": 0,
                    "callback_count": 0,
                    "padding_count": 0,
                }
                target = (
                    absorption_rows
                    if view_name == "fixed_graph4"
                    else residual_rows
                )
                target.append(common)
                matched_rows.append({
                    **common,
                    "contrast": "Dense_TopM_vs_Dense_Top(M-4)+Graph4_vs_additive",
                })
                for row in graph_selected:
                    qid = str(row["query_id"])
                    cid = str(row["selected_candidate_id"])
                    rid = str(row["replaced_candidate_id"])
                    action_rows.append({
                        "backend": backend,
                        "depth": depth,
                        "graph_view": view_name,
                        "repeat": row["repeat"],
                        "query_id": qid,
                        "candidate_id": cid,
                        "replaced_candidate_id": rid,
                        "candidate_utility": registry[(qid, cid)]["utility"],
                        "replaced_utility": registry[(qid, rid)]["utility"],
                        "raw_reward": row["raw_reward"],
                        "successful_action": row["successful_action"],
                        "harmful_action": row["harmful_action"],
                        "predicted_margin": row["predicted_margin"],
                    })
                print(json.dumps({
                    "backend": backend,
                    "depth": depth,
                    "view": view_name,
                    "oracle_marginal": common["oracle_marginal_vs_dense"],
                    "realised_marginal": common["realised_marginal_vs_dense"],
                }), flush=True)

    write_csv(out_dir / "m50_graph_absorption.csv", absorption_rows)
    write_csv(out_dir / "m50_graph_residual_value.csv", residual_rows)
    write_csv(out_dir / "m50_graph_matched_budget.csv", matched_rows)
    write_csv(out_dir / "m50_similarity_utility_actions.csv", action_rows)
    write_jsonl(out_dir / "m50_graph_oof_actions.jsonl", all_oof)
    write_json(out_dir / "nested_fold_audit.json", fold_audit)
    outputs = sorted(path for path in out_dir.iterdir() if path.is_file())
    write_json(out_dir / "manifest.json", {
        "schema": "dev100-m50-graph-frontier-analysis-v1",
        "created_utc": utc_now(),
        "status": "GRAPH_M50_FRONTIER_COMPLETE",
        "development_queries": 100,
        "depths": list(DEPTHS),
        "backends": list(BACKENDS),
        "graph_views": ["fixed_graph4", "residual_graph4"],
        "selector": "same Report91 Direct Huber source-blind nested OOF",
        "fixed_graph_provenance": "strict native Graph-practical4",
        "residual_graph_provenance": "strict native Graph4(M) outside Dense Top-M",
        "fallback_count": 0,
        "callback_count": 0,
        "padding_count": 0,
        "frozen_test_read": False,
        "m100_analysed": False,
        "remaining_development298_accessed": False,
        "input_hashes": {
            str(registry_path.relative_to(root)): sha256(registry_path),
            str((preflight / "manifest.json").relative_to(root)): sha256(
                preflight / "manifest.json"
            ),
            str(split_path.relative_to(root)): sha256(split_path),
            str((dense_dir / "manifest.json").relative_to(root)): sha256(
                dense_dir / "manifest.json"
            ),
        },
        "output_hashes": {path.name: sha256(path) for path in outputs},
    })


if __name__ == "__main__":
    main()
