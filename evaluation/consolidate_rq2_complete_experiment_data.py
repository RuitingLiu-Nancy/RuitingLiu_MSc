#!/usr/bin/env python3
"""Consolidate the frozen RQ2 experiment artifacts into one factual data report.

This script performs local joins and deterministic query-level bootstrap
recomputations only.  It does not call external providers, read hidden
community responses for retrieval, or modify thesis sources.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/rq2_complete_experiment_data_v1"
REPORT = ROOT / "RQ2_COMPLETE_EXPERIMENT_DATA.md"
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260805
FINAL_K = 8


def rq2_read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rq2_read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rq2_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rq2_mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def rq2_fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        if math.isnan(float(value)):
            return "—"
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):.{digits}f}"


def rq2_pct(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{100 * float(value):.1f}%"


def rq2_ci_text(low, high) -> str:
    if low is None or high is None:
        return "—"
    if any(math.isnan(float(x)) for x in (low, high)):
        return "—"
    return f"[{float(low):.4f}, {float(high):.4f}]"


def rq2_markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    headers = [label for _, label in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [str(row.get(key, "—")).replace("\n", " ").replace("|", "\\|") for key, _ in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def rq2_bootstrap_indices(n: int, seed: int = BOOTSTRAP_SEED, dtype=np.int64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n), dtype=dtype)


def rq2_paired_summary(values: list[float], indices: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    draws = arr[indices].mean(axis=1)
    return float(arr.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def rq2_oracle(ids: Iterable[str], qid: str, utility: dict[tuple[str, str], float]) -> float:
    unique = list(dict.fromkeys(str(candidate_id) for candidate_id in ids))
    values = [utility[(qid, candidate_id)] for candidate_id in unique]
    if len(values) < FINAL_K:
        raise ValueError(f"{qid}: pool has fewer than {FINAL_K} judged candidates")
    return rq2_mean(sorted(values, reverse=True)[:FINAL_K])


def rq2_write_csv(name: str, rows: list[dict]) -> Path:
    path = OUT / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def rq2_load_rankings(path: Path) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, list[dict]]]]:
    rows_by_backend: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rq2_read_jsonl(path):
        rows_by_backend[str(row["backend"])][str(row["query_id"])].append(row)
    ids: dict[str, dict[str, list[str]]] = {}
    rows_out: dict[str, dict[str, list[dict]]] = {}
    for backend, by_query in rows_by_backend.items():
        ids[backend] = {}
        rows_out[backend] = {}
        for qid, rows in by_query.items():
            ordered = sorted(rows, key=lambda item: int(item["rank"]))
            ids[backend][qid] = [str(item["comment_id"]) for item in ordered]
            rows_out[backend][qid] = ordered
    return ids, rows_out


def rq2_load_run(path: Path) -> dict[str, list[str]]:
    output = {}
    for row in rq2_read_jsonl(path):
        qid = str(row["query_id"])
        if "retrieved_titles" in row:
            output[qid] = [str(value) for value in row["retrieved_titles"]]
        else:
            output.setdefault(qid, []).append(str(row["comment_id"]))
    return output


def rq2_top_added(run: dict[str, list[str]], dense8: dict[str, list[str]], qids: list[str], quota: int = 4) -> dict[str, list[str]]:
    result = {}
    for qid in qids:
        seen = set(dense8[qid])
        selected = []
        for candidate_id in run[qid]:
            if candidate_id in seen:
                continue
            selected.append(candidate_id)
            seen.add(candidate_id)
            if len(selected) == quota:
                break
        result[qid] = selected
    return result


def rq2_coverage_row(label: str, pairs: set[tuple[str, str]], utility: dict[tuple[str, str], float]) -> dict:
    judged = len(pairs & set(utility))
    total = len(pairs)
    return {
        "variant": label,
        "pairs": total,
        "already": judged,
        "new": 0,
        "missing": total - judged,
        "coverage": judged / total if total else math.nan,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    query_path = ROOT / "out/development300_analysis_inputs_v1/development300_queries_normalized.json"
    admin_path = ROOT / "out/development300_stratified_v1/development300_queries_ADMIN.csv"
    corpus_path = ROOT / "out/hipporag_official_adapter/adhd_peer_support_validation_corpus.json"
    corpus_map_path = ROOT / "out/expanded_graph_rebuild/unified_extract_input_1618.csv"
    dense_path = ROOT / "out/development300_m50_preflight_v1/dense_m50_memberships.jsonl"
    graph_path = ROOT / "out/development300_m50_preflight_v1/strict_fixed_graph4.jsonl"
    registry_path = ROOT / "out/development300_m50_utility_judging_v2/complete/utility_registry_coverage_complete.jsonl"
    selector_dir = ROOT / "experiments/selection_action_space_repair_dev300_v2"
    set_dir = ROOT / "experiments/rq2b_set_correspondence_dev100_v1"

    queries = rq2_read_json(query_path)
    qids = sorted(str(row["id"]) for row in queries)
    if len(qids) != 300 or len(set(qids)) != 300:
        raise ValueError("Development300 query identity changed")
    qindex = {qid: index for index, qid in enumerate(qids)}
    indices = rq2_bootstrap_indices(len(qids), dtype=np.int32)
    if hashlib.sha256(indices.tobytes()).hexdigest() != "f1804695368b8bbea9d70bb9e39b4764ac22c4fdc552c40b2f7308d985f09b8a":
        raise ValueError("Development300 bootstrap index identity changed")

    admin = pd.read_csv(admin_path, dtype=str)
    if set(admin["query_id"]) != set(qids):
        raise ValueError("Development300 admin query set mismatch")
    component_counts = admin["development300_component"].value_counts().to_dict()
    need_counts = admin["llm_single_multi_label"].value_counts().to_dict()
    query_post = dict(zip(admin["query_id"], admin["post_id"]))

    registry_rows = rq2_read_jsonl(registry_path)
    utility = {(str(row["query_id"]), str(row["comment_id"])): float(row["utility"]) for row in registry_rows}
    if len(utility) != 36975:
        raise ValueError(f"utility registry identity changed: {len(utility)}")

    rankings, ranking_rows = rq2_load_rankings(dense_path)
    for backend in ("minilm", "e5"):
        if set(rankings[backend]) != set(qids) or any(len(rankings[backend][qid]) != 50 for qid in qids):
            raise ValueError(f"{backend} D50 identity changed")

    fixed_rows = rq2_read_jsonl(graph_path)
    fixed: dict[str, list[str]] = defaultdict(list)
    for row in sorted(fixed_rows, key=lambda value: (str(value["query_id"]), int(value["source_rank"]))):
        if not row["native_graph"] or row["fallback_used"] or row["callback_used"] or row["padding_used"]:
            raise ValueError("fixed Graph4 strict provenance changed")
        fixed[str(row["query_id"])].append(str(row["candidate_id"]))
    if set(fixed) != set(qids) or any(len(fixed[qid]) != 4 for qid in qids):
        raise ValueError("fixed Graph4 identity changed")

    # Table A: dense candidate availability.
    table_a = []
    dense_oracles: dict[tuple[str, int], dict[str, float]] = {}
    raw_means: dict[str, float] = {}
    for backend in ("minilm", "e5"):
        per_depth = {}
        for depth in (8, 12, 20, 50):
            per_query = {qid: rq2_oracle(rankings[backend][qid][:depth], qid, utility) for qid in qids}
            dense_oracles[(backend, depth)] = per_query
            per_depth[depth] = per_query
        raw_means[backend] = rq2_mean(per_depth[8].values())
        for depth in (8, 12, 20, 50):
            pool_pairs = [(qid, cid) for qid in qids for cid in rankings[backend][qid][:depth]]
            deltas = [per_depth[depth][qid] - per_depth[8][qid] for qid in qids]
            delta, low, high = rq2_paired_summary(deltas, indices)
            useful = sum(utility[pair] >= 4.0 for pair in pool_pairs)
            table_a.append({
                "backend": backend,
                "pool": f"D{depth}",
                "candidates_per_query": depth,
                "mean_pool_u": rq2_mean(utility[pair] for pair in pool_pairs),
                "oracle_u8": rq2_mean(per_depth[depth].values()),
                "delta": delta,
                "ci_low": low,
                "ci_high": high,
                "useful_count": useful,
                "useful_fraction": useful / len(pool_pairs),
            })
    rq2_write_csv("table_a_dense_candidate_availability.csv", table_a)

    # Table B: fixed Graph4 candidate frontier.
    table_b = []
    graph_oracles: dict[tuple[str, int], dict[str, float]] = {}
    for backend in ("minilm", "e5"):
        for depth in (8, 12, 20, 50):
            union_oracle = {}
            added_by_query = {}
            for qid in qids:
                dense_ids = rankings[backend][qid][:depth]
                added = [candidate_id for candidate_id in fixed[qid] if candidate_id not in set(dense_ids)]
                added_by_query[qid] = added
                union_oracle[qid] = rq2_oracle([*dense_ids, *fixed[qid]], qid, utility)
            graph_oracles[(backend, depth)] = union_oracle
            deltas = [union_oracle[qid] - dense_oracles[(backend, depth)][qid] for qid in qids]
            delta, low, high = rq2_paired_summary(deltas, indices)
            added_pairs = [(qid, cid) for qid in qids for cid in added_by_query[qid]]
            table_b.append({
                "backend": backend,
                "M": depth,
                "graph_rows": len(fixed_rows),
                "dense_oracle": rq2_mean(dense_oracles[(backend, depth)].values()),
                "union_oracle": rq2_mean(union_oracle.values()),
                "marginal": delta,
                "ci_low": low,
                "ci_high": high,
                "unique_fraction": len(added_pairs) / len(fixed_rows),
                "mean_added_u": rq2_mean(utility[pair] for pair in added_pairs),
                "mean_added_candidates": len(added_pairs) / len(qids),
            })
    rq2_write_csv("table_b_principal_graph_frontier.csv", table_b)

    # Table E: exact reuse of the existing label-blind matched-budget rule.
    table_e = []
    matched_memberships = []
    for backend in ("minilm", "e5"):
        for depth in (12, 20, 50):
            matched_oracle = {}
            final_counts = []
            graph_unique = []
            for qid in qids:
                dense_m = rankings[backend][qid][:depth]
                selected = list(dict.fromkeys([*dense_m[: depth - 4], *fixed[qid]]))
                for candidate_id in dense_m:
                    if len(selected) >= depth:
                        break
                    if candidate_id not in selected:
                        selected.append(candidate_id)
                if len(selected) != depth or not set(rankings[backend][qid][:8]).issubset(selected):
                    raise ValueError(f"matched-budget invariant failed: {backend}/{depth}/{qid}")
                final_counts.append(len(selected))
                graph_unique.append(sum(candidate_id not in set(dense_m[: depth - 4]) for candidate_id in fixed[qid]))
                matched_oracle[qid] = rq2_oracle(selected, qid, utility)
                matched_memberships.extend({
                    "backend": backend,
                    "M": depth,
                    "query_id": qid,
                    "candidate_id": candidate_id,
                    "rank": rank,
                    "fixed_graph4_member": candidate_id in set(fixed[qid]),
                } for rank, candidate_id in enumerate(selected, start=1))
            deltas = [matched_oracle[qid] - dense_oracles[(backend, depth)][qid] for qid in qids]
            delta, low, high = rq2_paired_summary(deltas, indices)
            table_e.append({
                "backend": backend,
                "M": depth,
                "all_dense_pool": f"D{depth}",
                "matched_graph_pool": f"D{depth-4}+G4 with D{depth} backfill",
                "dense_oracle": rq2_mean(dense_oracles[(backend, depth)].values()),
                "matched_oracle": rq2_mean(matched_oracle.values()),
                "delta": delta,
                "ci_low": low,
                "ci_high": high,
                "graph_unique_fraction": rq2_mean(graph_unique) / 4,
                "final_candidate_count": rq2_mean(final_counts),
            })
    rq2_write_csv("table_e_matched_budget_oracle.csv", table_e)
    rq2_write_csv("matched_budget_memberships.csv", matched_memberships)

    # Tables F/G: saved Development300 selector results.
    dense_summary = pd.read_csv(selector_dir / "dense_summary.csv")
    dense_cells = dense_summary[
        (dense_summary["stratum"] == "all")
        & (dense_summary["dense_depth"].isin([12, 20, 50]))
        & (dense_summary["replacement_budget"].isin([1, 2, 4, 8]))
    ].copy()
    table_f = []
    table_g = []
    scorer_labels = {"candidate_huber": "Huber", "candidate_small_mlp": "Small MLP"}
    for _, row in dense_cells.sort_values(["backend", "dense_depth", "scorer", "replacement_budget"]).iterrows():
        table_f.append({
            "backend": row.backend,
            "M": int(row.dense_depth),
            "scorer": scorer_labels[row.scorer],
            "r": int(row.replacement_budget),
            "realised_delta": row.mean_realised_gain,
            "ci_low": row.realised_gain_ci_low,
            "ci_high": row.realised_gain_ci_high,
            "absolute_u8": row.mean_selected_utility_at8,
            "oracle_headroom": row.mean_candidate_access_headroom,
            "conversion": row.full_pool_conversion,
            "harm_rate": row.harmful_query_rate,
            "wins": int(row.wins),
            "ties": int(row.ties),
            "losses": int(row.losses),
        })
        table_g.append({
            "backend": row.backend,
            "M": int(row.dense_depth),
            "scorer": scorer_labels[row.scorer],
            "r": int(row.replacement_budget),
            "raw_u8": raw_means[row.backend],
            "oracle_absolute_u8": raw_means[row.backend] + row.mean_candidate_access_headroom,
            "realised_absolute_u8": row.mean_selected_utility_at8,
            "access_delta": row.mean_candidate_access_headroom,
            "realised_delta": row.mean_realised_gain,
            "unconverted_gap": row.mean_candidate_access_headroom - row.mean_realised_gain,
            "conversion": row.full_pool_conversion,
        })
    rq2_write_csv("table_f_utility_conversion.csv", table_f)
    rq2_write_csv("table_g_oracle_realised_conversion.csv", table_g)

    # Table H: saved Graph conversion cells, with direct Graph marginal and harm.
    graph_summary = pd.read_csv(selector_dir / "graph_summary.csv")
    graph_cells = graph_summary[
        (graph_summary["stratum"] == "all")
        & (graph_summary["dense_depth"].isin([8, 12]))
        & (graph_summary["replacement_budget"].isin([1, 8]))
    ].copy()
    table_h = []
    for _, row in graph_cells.sort_values(["backend", "dense_depth", "scorer", "replacement_budget"]).iterrows():
        oracle_marginal = next(
            item["marginal"] for item in table_b
            if item["backend"] == row.backend and item["M"] == int(row.dense_depth)
        )
        realised = float(row.mean_graph_policy_marginal_vs_dense)
        table_h.append({
            "backend": row.backend,
            "M": int(row.dense_depth),
            "scorer": scorer_labels[row.scorer],
            "capacity": "conservative r=1" if int(row.replacement_budget) == 1 else "unrestricted r=8",
            "oracle_graph_marginal": oracle_marginal,
            "realised_graph_marginal": realised,
            "ci_low": row.graph_policy_marginal_ci_low,
            "ci_high": row.graph_policy_marginal_ci_high,
            "conversion": realised / oracle_marginal if oracle_marginal else math.nan,
            "entrant_precision": row.graph_entrant_precision,
            "harm_rate": row.graph_marginal_losses / row.queries,
        })
    rq2_write_csv("table_h_graph_conversion.csv", table_h)

    # Table I: saved set-level results; recompute every method-vs-raw CI from per-query data.
    set_summary = pd.read_csv(set_dir / "rq2b_c_method_summary.csv")
    set_per_query = pd.read_csv(set_dir / "rq2b_c_per_query.csv")
    set_qids = sorted(set_per_query["query_id"].astype(str).unique())
    set_indices = rq2_bootstrap_indices(len(set_qids), seed=20260810)
    set_index_hash = hashlib.sha256(
        json.dumps(set_indices.tolist(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if set_index_hash != "e32e355898aeb8e687857ce70ae845fa481f7ef55620cadcf9139e95fa2a34d3":
        raise ValueError("Development100 set-level bootstrap index identity changed")
    set_metrics = ["utility_at8", "cra_at8", "rcc_at8", "bialign_f1_at8", "best_align_at8"]
    raw_set = set_per_query[set_per_query.method == "raw_dense8"].set_index("query_id").loc[set_qids]
    table_i = []
    for method in ["raw_dense8", "one_swap", "two_swap", "direct", "mmr_r2"]:
        current = set_per_query[set_per_query.method == method].set_index("query_id").loc[set_qids]
        row_out = {"method": method, "N": len(set_qids)}
        for metric in set_metrics:
            delta_values = (current[metric] - raw_set[metric]).to_numpy(float)
            delta, low, high = rq2_paired_summary(delta_values.tolist(), set_indices)
            row_out[f"mean_{metric}"] = float(current[metric].mean())
            row_out[f"delta_{metric}"] = delta
            row_out[f"ci_low_{metric}"] = low
            row_out[f"ci_high_{metric}"] = high
        table_i.append(row_out)
    rq2_write_csv("table_i_selected_set_dev100.csv", table_i)

    # Table J: frozen independent Test200 contrasts.
    test_path = ROOT / "out/confirmatory_test200_v1/analysis_v1_output_union/paired_contrasts.json"
    table_j = rq2_read_json(test_path)
    rq2_write_csv("table_j_independent_test200.csv", table_j)

    # Development300 BM25 local retrieval and strict same-thread audit.
    bm25_path = OUT / "bm25_development300_normalized.jsonl"
    bm25 = rq2_load_run(bm25_path)
    if set(bm25) != set(qids) or any(len(bm25[qid]) != 100 for qid in qids):
        raise ValueError("normalized BM25 Development300 identity changed")
    corpus_map = pd.read_csv(corpus_map_path, dtype=str)
    candidate_post = dict(zip(corpus_map["comment_id"], corpus_map["post_id"]))
    bm25_added = {}
    same_thread_leaks = {}
    for backend in ("minilm", "e5"):
        bm25_added[backend] = rq2_top_added(bm25, {qid: rankings[backend][qid][:8] for qid in qids}, qids)
        unmapped = [(qid, cid) for qid in qids for cid in bm25_added[backend][qid] if cid not in candidate_post]
        if unmapped:
            raise ValueError(f"BM25 candidates missing corpus post mapping for {backend}: {len(unmapped)}")
        leaks = [(qid, cid) for qid in qids for cid in bm25_added[backend][qid] if candidate_post.get(cid) == query_post[qid]]
        same_thread_leaks[backend] = leaks
        if leaks:
            raise ValueError(f"BM25 same-thread leakage detected for {backend}: {len(leaks)}")
    bm25_reproduction = {}
    for label, prior_path in {
        "nested_dev100": ROOT / "out/dev100_v2_versioned_runs/runs/bm25s.jsonl",
        "frozen_development200_shared": ROOT / "out/development200_shortlist_v1/runs/bm25s.jsonl",
    }.items():
        prior = rq2_load_run(prior_path)
        shared = sorted(set(bm25) & set(prior))
        exact = sum(bm25[qid][:100] == prior[qid][:100] for qid in shared)
        if exact != len(shared):
            raise ValueError(f"BM25 reproduction failed: {label} {exact}/{len(shared)}")
        bm25_reproduction[label] = {"shared_queries": len(shared), "top100_exact_queries": exact}

    # Full Development300 no-recognition route from nested Dev100 + additional200.
    no_rec_legacy = rq2_load_run(ROOT / "out/development200_sbert_official_v1/no_recognition.jsonl")
    no_rec_additional = rq2_load_run(ROOT / "out/development300_sbert_official_additional200_v1/no_recognition.jsonl")
    no_rec = {qid: (no_rec_additional if qid in no_rec_additional else no_rec_legacy)[qid] for qid in qids}
    no_rec_added = {
        backend: rq2_top_added(no_rec, {qid: rankings[backend][qid][:8] for qid in qids}, qids)
        for backend in ("minilm", "e5")
    }
    no_rec_same_thread_leaks = {
        backend: [(qid, cid) for qid in qids for cid in no_rec_added[backend][qid] if candidate_post.get(cid) == query_post[qid]]
        for backend in ("minilm", "e5")
    }
    no_rec_unmapped = {
        backend: [(qid, cid) for qid in qids for cid in no_rec_added[backend][qid] if cid not in candidate_post]
        for backend in ("minilm", "e5")
    }
    if any(no_rec_unmapped.values()):
        raise ValueError("no-recognition G4 contains candidates without corpus post mapping")
    if any(no_rec_same_thread_leaks.values()):
        raise ValueError("no-recognition G4 contains source-thread candidates")

    no_rec_metrics = {}
    for backend in ("minilm", "e5"):
        no_rec_oracle = {
            qid: rq2_oracle([*rankings[backend][qid][:8], *no_rec_added[backend][qid]], qid, utility)
            for qid in qids
        }
        delta_d8 = [no_rec_oracle[qid] - dense_oracles[(backend, 8)][qid] for qid in qids]
        delta_principal = [no_rec_oracle[qid] - graph_oracles[(backend, 8)][qid] for qid in qids]
        d8_mean, d8_low, d8_high = rq2_paired_summary(delta_d8, indices)
        principal_mean, principal_low, principal_high = rq2_paired_summary(delta_principal, indices)
        pairs = {(qid, cid) for qid in qids for cid in no_rec_added[backend][qid]}
        if not pairs.issubset(utility):
            raise ValueError(f"no-recognition G4 unexpectedly lacks utility coverage: {backend}")
        no_rec_metrics[backend] = {
            "oracle_u8": rq2_mean(no_rec_oracle.values()),
            "delta_d8": d8_mean,
            "ci_low": d8_low,
            "ci_high": d8_high,
            "delta_principal": principal_mean,
            "principal_ci_low": principal_low,
            "principal_ci_high": principal_high,
            "mean_added_u": rq2_mean(utility[pair] for pair in pairs),
            "mean_added_candidates": rq2_mean(len(no_rec_added[backend][qid]) for qid in qids),
            "unique_fraction": len(pairs) / 1200,
        }

    # Historical Development200 Graph variants: coverage only on the 172-query
    # intersection with Development300.  Their normalized rows do not retain
    # enough provenance to certify strict-native G4 on the full cohort.
    historical_variant_paths = {
        "Historical recognition/static G4 — shared N=172": ROOT / "out/development200_shortlist_v1/runs/official_static.jsonl",
        "Historical node+hub G4 — shared N=172": ROOT / "out/development200_shortlist_v1/runs/official_node_hub.jsonl",
        "Historical local-PCST G4 — shared N=172": ROOT / "out/development200_shortlist_v1/runs/official_local_pcst.jsonl",
    }
    historical_variant_added = {}
    for label, path in historical_variant_paths.items():
        run = rq2_load_run(path)
        shared_qids = sorted(set(qids) & set(run))
        if len(shared_qids) != 172:
            raise ValueError(f"historical Development200 intersection changed: {label}")
        historical_variant_added[label] = {
            backend: rq2_top_added(run, {qid: rankings[backend][qid][:8] for qid in shared_qids}, shared_qids)
            for backend in ("minilm", "e5")
        }

    # Table K: utility registry joins; no external calls or imputation.
    formal_pairs = {(str(row["query_id"]), str(row["comment_id"])) for row in rq2_read_jsonl(ROOT / "out/development300_m50_preflight_v1/formal_union_manifest.jsonl")}
    fixed_pairs = {(qid, cid) for qid in qids for cid in fixed[qid]}
    if not fixed_pairs.issubset(formal_pairs):
        raise ValueError("maintained fixed Graph4 is outside the formal union")
    for backend in ("minilm", "e5"):
        route_pairs = {(qid, cid) for qid in qids for cid in no_rec_added[backend][qid]}
        if not route_pairs.issubset(formal_pairs):
            raise ValueError(f"no-recognition G4 is outside the formal union: {backend}")
    table_k = [
        rq2_coverage_row("Development300 formal union", formal_pairs, utility),
        rq2_coverage_row("Maintained fixed Graph4", fixed_pairs, utility),
    ]
    for label, mapping in [("No-recognition G4", no_rec_added), ("BM25 B4", bm25_added)]:
        union_pairs = set()
        for backend in ("minilm", "e5"):
            pairs = {(qid, cid) for qid in qids for cid in mapping[backend][qid]}
            union_pairs |= pairs
            table_k.append(rq2_coverage_row(f"{label} — {backend}", pairs, utility))
        table_k.append(rq2_coverage_row(f"{label} — backend union", union_pairs, utility))
    for label, mapping in historical_variant_added.items():
        union_pairs = set()
        for backend in ("minilm", "e5"):
            pairs = {(qid, cid) for qid, ids in mapping[backend].items() for cid in ids}
            union_pairs |= pairs
            table_k.append(rq2_coverage_row(f"{label} — {backend}", pairs, utility))
        table_k.append(rq2_coverage_row(f"{label} — backend union", union_pairs, utility))
    rq2_write_csv("table_k_utility_coverage.csv", table_k)

    # Table C records only an actual full comparable row; other requested variants retain factual status.
    table_c = []
    for backend in ("minilm", "e5"):
        b = next(row for row in table_b if row["backend"] == backend and row["M"] == 8)
        table_c.append({
            "backend": backend,
            "variant": "Maintained fixed Graph4 (round-robin no-recognition + fact-only-no-recognition PPR)",
            "status": "COMPLETE",
            "valid_queries": 300,
            "mean_added_candidates": b["mean_added_candidates"],
            "unique_fraction": b["unique_fraction"],
            "mean_added_u": b["mean_added_u"],
            "oracle_u8": b["union_oracle"],
            "delta_d8": b["marginal"],
            "ci_low": b["ci_low"],
            "ci_high": b["ci_high"],
            "delta_principal": 0.0,
            "principal_ci_low": 0.0,
            "principal_ci_high": 0.0,
            "reason": "Actual maintained strict Graph4; generated once and reused for both anchors.",
        })
        no_rec_row = no_rec_metrics[backend]
        table_c.append({
            "backend": backend,
            "variant": "No recognition — single route",
            "status": "COMPLETE",
            "valid_queries": 300,
            "mean_added_candidates": no_rec_row["mean_added_candidates"],
            "unique_fraction": no_rec_row["unique_fraction"],
            "mean_added_u": no_rec_row["mean_added_u"],
            "oracle_u8": no_rec_row["oracle_u8"],
            "delta_d8": no_rec_row["delta_d8"],
            "ci_low": no_rec_row["ci_low"],
            "ci_high": no_rec_row["ci_high"],
            "delta_principal": no_rec_row["delta_principal"],
            "principal_ci_low": no_rec_row["principal_ci_low"],
            "principal_ci_high": no_rec_row["principal_ci_high"],
            "reason": "Strict single no-recognition PPR route; 1,200/1,200 backend-specific pairs judged.",
        })
    partial_variants = [
        ("Requested recognition-entry static PPR", 172, "Frozen Development200 run has 172/300 shared queries (128 missing; 28 run queries outside Dev300); not the maintained Graph4 and no full strict G4 artifact."),
        ("Structured reformulation (original + structured)", 100, "Frozen only on nested Development100; entry on 89/100 queries."),
        ("Corrected PPR (node + hub)", 172, "Frozen Development200 run has 172/300 shared queries (128 missing; 28 outside Dev300); normalized rows do not certify full strict-native provenance."),
        ("Local PCST", 172, "Frozen Development200 run has 172/300 shared queries (128 missing; 28 outside Dev300); normalized rows do not certify full strict-native provenance."),
        ("Learned propagation (seed-only head)", 28, "Entered-query subset only; final manifest is NO_GO_LEARNED_EDGE_DIFFUSION and no deployment checkpoint."),
    ]
    for backend in ("minilm", "e5"):
        for variant, n, reason in partial_variants:
            table_c.append({
                "backend": backend, "variant": variant, "status": "PARTIAL", "valid_queries": n,
                "mean_added_candidates": math.nan, "unique_fraction": math.nan, "mean_added_u": math.nan,
                "oracle_u8": math.nan, "delta_d8": math.nan, "ci_low": math.nan, "ci_high": math.nan,
                "delta_principal": math.nan, "principal_ci_low": math.nan, "principal_ci_high": math.nan,
                "reason": reason,
            })
    rq2_write_csv("table_c_graph_variant_comparison.csv", table_c)

    # Table D cannot be completed without the frozen utility labels for new BM25 pairs.
    table_d = []
    bm25_complete_queries = {}
    for backend in ("minilm", "e5"):
        graph = next(row for row in table_b if row["backend"] == backend and row["M"] == 8)
        bm25_pairs = {(qid, cid) for qid in qids for cid in bm25_added[backend][qid]}
        bm25_complete_queries[backend] = sum(
            all((qid, cid) in utility for cid in bm25_added[backend][qid]) for qid in qids
        )
        table_d.extend([
            {
                "backend": backend, "route": "Principal Graph4", "status": "COMPLETE",
                "oracle_u8": graph["union_oracle"], "delta_d8": graph["marginal"],
                "ci_low": graph["ci_low"], "ci_high": graph["ci_high"],
                "unique_fraction": graph["unique_fraction"], "mean_added_u": graph["mean_added_u"],
                "delta_graph4": 0.0, "graph_ci_low": 0.0, "graph_ci_high": 0.0,
            },
            {
                "backend": backend, "route": "BM25 B4", "status": f"BLOCKED: utility-complete queries {bm25_complete_queries[backend]}/300; {len(bm25_pairs - set(utility))} missing pairs",
                "oracle_u8": math.nan, "delta_d8": math.nan, "ci_low": math.nan, "ci_high": math.nan,
                "unique_fraction": len(bm25_pairs) / 1200, "mean_added_u": math.nan,
                "delta_graph4": math.nan, "graph_ci_low": math.nan, "graph_ci_high": math.nan,
            },
        ])
    rq2_write_csv("table_d_bm25_vs_graph.csv", table_d)

    # Intermediate Graph diagnostics.
    gates = rq2_read_json(ROOT / "out/official_round2_dev100/round2_gates.json")["gates"]
    rewrite = rq2_read_json(ROOT / "out/query_rewrite_graph_entry_dev100_v2/label_blind_analysis/rewrite_entry_report.json")
    static_repro = rq2_read_json(ROOT / "out/learned_diffusion_dev100_v1/static_ppr_reproduction_report.json")
    learned = rq2_read_json(ROOT / "out/learned_diffusion_dev100_v1/seed_only_cv_report.json")
    learned_pairs = rq2_read_json(ROOT / "out/learned_diffusion_dev100_v1/cv_pairwise_comparisons.json")
    diagnostics = []
    for method in ["official_node_hub", "official_local_pcst", "dense_bridge_2hop", "spread_node_hub_3hop", "pcst_component_coverage"]:
        row = gates[method]
        diagnostics.append({
            "dataset": "Development100", "method": method, "scope": "candidate distribution",
            "metric": "Exclusive@20 vs official static", "value": row["mean_exclusive_at20_vs_official"],
            "detail": f"Jaccard@20={row['jaccard20_vs_official']:.4f}",
        })
    for method in ["structured", "original_plus_structured"]:
        row = rewrite["methods"][method]
        diagnostics.extend([
            {"dataset": "Development100", "method": method, "scope": "entry", "metric": "entry rate", "value": row["entry_rate_all_dev100"], "detail": f"entry={row['entry_queries']}/100; available={row['available_queries']}"},
            {"dataset": "Development100", "method": method, "scope": "entry", "metric": "mean graph seed count", "value": row["mean_graph_seed_count_available"], "detail": "available queries"},
        ])
    diagnostics.extend([
        {"dataset": "Entered Development100 subset", "method": "static PPR reproduction", "scope": f"N={static_repro['queries']}", "metric": "mean Top100 Jaccard", "value": static_repro["mean_top100_jaccard"], "detail": f"minimum Top100 score Spearman={static_repro['minimum_top100_score_spearman']:.4f}"},
        {"dataset": "Entered Development100 subset", "method": "learned seed + head vs static PPR", "scope": "N=28; judged-only", "metric": "nDCG@3 delta", "value": next(row["mean_delta"] for row in learned_pairs if row.get("left") == "learned_seed_head" and row.get("right") == "official_static" and row["metric"] == "judged_only_ndcg_at3"), "detail": "see frozen paired-comparison artifact"},
    ])
    rq2_write_csv("graph_intermediate_diagnostics.csv", diagnostics)

    # Render one factual Markdown report.
    a_rows = [{
        "backend": row["backend"], "pool": row["pool"], "N": 300,
        "mean": rq2_fmt(row["mean_pool_u"]), "oracle": rq2_fmt(row["oracle_u8"]), "delta": rq2_fmt(row["delta"]),
        "ci": rq2_ci_text(row["ci_low"], row["ci_high"]),
        "useful": f"{row['useful_count']}/{300 * row['candidates_per_query']} ({rq2_pct(row['useful_fraction'])})",
    } for row in table_a]
    b_rows = [{
        "backend": row["backend"], "M": row["M"], "N": 300,
        "dense": rq2_fmt(row["dense_oracle"]), "union": rq2_fmt(row["union_oracle"]), "delta": rq2_fmt(row["marginal"]),
        "ci": rq2_ci_text(row["ci_low"], row["ci_high"]), "unique": rq2_pct(row["unique_fraction"]),
        "added": rq2_fmt(row["mean_added_u"]), "count": rq2_fmt(row["mean_added_candidates"], 3),
    } for row in table_b]
    c_rows = [{
        "backend": row["backend"], "variant": row["variant"], "status": row["status"], "N": row["valid_queries"],
        "count": rq2_fmt(row["mean_added_candidates"], 3), "unique": rq2_pct(row["unique_fraction"]),
        "added": rq2_fmt(row["mean_added_u"]), "oracle": rq2_fmt(row["oracle_u8"]), "delta": rq2_fmt(row["delta_d8"]),
        "ci": rq2_ci_text(row["ci_low"], row["ci_high"]), "principal": rq2_fmt(row["delta_principal"]),
        "principal_ci": rq2_ci_text(row["principal_ci_low"], row["principal_ci_high"]), "reason": row["reason"],
    } for row in table_c]
    d_rows = [{
        "backend": row["backend"], "route": row["route"], "status": row["status"], "oracle": rq2_fmt(row["oracle_u8"]),
        "delta": rq2_fmt(row["delta_d8"]), "ci": rq2_ci_text(row["ci_low"], row["ci_high"]),
        "unique": rq2_pct(row["unique_fraction"]), "added": rq2_fmt(row["mean_added_u"]),
        "graph": rq2_fmt(row["delta_graph4"]), "graph_ci": rq2_ci_text(row["graph_ci_low"], row["graph_ci_high"]),
    } for row in table_d]
    e_rows = [{
        "backend": row["backend"], "M": row["M"], "dense_pool": row["all_dense_pool"], "matched_pool": row["matched_graph_pool"],
        "dense": rq2_fmt(row["dense_oracle"]), "matched": rq2_fmt(row["matched_oracle"]), "delta": rq2_fmt(row["delta"]),
        "ci": rq2_ci_text(row["ci_low"], row["ci_high"]), "unique": rq2_pct(row["graph_unique_fraction"]),
        "count": rq2_fmt(row["final_candidate_count"], 1),
    } for row in table_e]
    f_rows = [{
        "backend": row["backend"], "M": row["M"], "scorer": row["scorer"], "r": row["r"],
        "delta": rq2_fmt(row["realised_delta"]), "ci": rq2_ci_text(row["ci_low"], row["ci_high"]),
        "absolute": rq2_fmt(row["absolute_u8"]), "oracle": rq2_fmt(row["oracle_headroom"]), "conversion": rq2_pct(row["conversion"]),
        "harm": rq2_pct(row["harm_rate"]), "wtl": f"{row['wins']}/{row['ties']}/{row['losses']}",
    } for row in table_f]
    g_rows = [{
        "backend": row["backend"], "M": row["M"], "scorer": row["scorer"], "r": row["r"],
        "raw": rq2_fmt(row["raw_u8"]), "oracle_abs": rq2_fmt(row["oracle_absolute_u8"]), "real_abs": rq2_fmt(row["realised_absolute_u8"]),
        "access": rq2_fmt(row["access_delta"]), "real": rq2_fmt(row["realised_delta"]), "gap": rq2_fmt(row["unconverted_gap"]),
        "conversion": rq2_pct(row["conversion"]),
    } for row in table_g]
    h_rows = [{
        "backend": row["backend"], "M": row["M"], "scorer": row["scorer"], "capacity": row["capacity"],
        "oracle": rq2_fmt(row["oracle_graph_marginal"]), "real": rq2_fmt(row["realised_graph_marginal"]),
        "ci": rq2_ci_text(row["ci_low"], row["ci_high"]), "conversion": rq2_pct(row["conversion"]),
        "precision": rq2_pct(row["entrant_precision"]), "harm": rq2_pct(row["harm_rate"]),
    } for row in table_h]
    method_labels = {"raw_dense8": "Raw", "one_swap": "One-swap", "two_swap": "Two-swap", "direct": "Direct", "mmr_r2": "MMR r=2"}
    i_rows = []
    for row in table_i:
        item = {"method": method_labels[row["method"]], "N": row["N"]}
        for key, short in [("utility_at8", "u"), ("rcc_at8", "rcc"), ("cra_at8", "cra"), ("bialign_f1_at8", "bi"), ("best_align_at8", "best")]:
            item[short] = f"{rq2_fmt(row[f'mean_{key}'])}; Δ {rq2_fmt(row[f'delta_{key}'])} {rq2_ci_text(row[f'ci_low_{key}'], row[f'ci_high_{key}'])}"
        i_rows.append(item)
    j_rows = [{
        "backend": row["backend"], "contrast": row["contrast"], "N": row["queries"],
        "delta": rq2_fmt(row["mean_delta_utility_at8"]), "ci": rq2_ci_text(*row["paired_query_bootstrap_95ci"]),
        "wtl": f"{row['wins']}/{row['ties']}/{row['losses']}",
    } for row in table_j]
    k_rows = [{
        "variant": row["variant"], "pairs": row["pairs"], "already": row["already"], "new": row["new"],
        "missing": row["missing"], "coverage": rq2_pct(row["coverage"]),
    } for row in table_k]
    diag_rows = [{"dataset": row["dataset"], "method": row["method"], "scope": row["scope"], "metric": row["metric"], "value": rq2_fmt(row["value"]), "detail": row["detail"]} for row in diagnostics]

    coverage_matrix = [
        {"experiment": "Dense frontier", "dataset": "Development300", "pool": "D8/D12/D20/D50", "outcome": "Oracle U@8", "existing": "Yes", "complete": "COMPLETE", "source": "selection_action_space_repair_dev300_v2 + frozen registry", "action": "Verified/recomputed"},
        {"experiment": "Maintained fixed Graph4 frontier", "dataset": "Development300", "pool": "D_M ∪ G4", "outcome": "Oracle U@8", "existing": "Yes", "complete": "COMPLETE", "source": "development300_m50_preflight_v1", "action": "Verified/recomputed"},
        {"experiment": "Requested recognition/static Graph", "dataset": "Development300", "pool": "D8 ∪ G4", "outcome": "Oracle U@8", "existing": "Partial", "complete": "PARTIAL N=172 raw overlap", "source": "heterogeneous historical Graph runs", "action": "Not merged"},
        {"experiment": "No-recognition single route", "dataset": "Development300", "pool": "D8 ∪ G4", "outcome": "Oracle U@8", "existing": "Yes", "complete": "COMPLETE", "source": "development200 + additional200 official runs", "action": "Verified/recomputed"},
        {"experiment": "Structured reformulation", "dataset": "Development100", "pool": "entry/ranked candidates", "outcome": "Entry diagnostics", "existing": "Yes", "complete": "PARTIAL N=100", "source": "query_rewrite_graph_entry_dev100_v2", "action": "Retained as diagnostic"},
        {"experiment": "Corrected PPR node+hub", "dataset": "Development300 target", "pool": "D8 ∪ G4", "outcome": "Oracle U@8", "existing": "Partial", "complete": "PARTIAL N=172", "source": "development200_shortlist_v1 + Dev100 diagnostics", "action": "Coverage audited; not pooled"},
        {"experiment": "Local PCST", "dataset": "Development300 target", "pool": "D8 ∪ G4", "outcome": "Oracle U@8", "existing": "Partial", "complete": "PARTIAL N=172", "source": "development200_shortlist_v1 + Dev100 diagnostics", "action": "Coverage audited; not pooled"},
        {"experiment": "Learned propagation", "dataset": "entered Dev100 subset", "pool": "local graph", "outcome": "judged-only nDCG@3", "existing": "Yes", "complete": "PARTIAL N=28 / NO_GO", "source": "learned_diffusion_dev100_v1", "action": "Retained as diagnostic"},
        {"experiment": "BM25 shallow control", "dataset": "Development300", "pool": "D8 ∪ B4", "outcome": "Oracle U@8", "existing": "New candidates", "complete": "BLOCKED: utility coverage", "source": "normalized BM25 Development300 run", "action": "Retrieval/audit complete"},
        {"experiment": "Matched-budget Oracle", "dataset": "Development300", "pool": "D_M vs D_(M-4) ∪ G4", "outcome": "Oracle U@8", "existing": "Reconstructable", "complete": "COMPLETE", "source": "frozen rankings/Graph4/registry", "action": "Newly recomputed"},
        {"experiment": "Dense utility conversion", "dataset": "Development300", "pool": "D12/D20/D50", "outcome": "realised U@8", "existing": "Yes", "complete": "COMPLETE", "source": "selection_action_space_repair_dev300_v2", "action": "Consolidated"},
        {"experiment": "Graph utility conversion", "dataset": "Development300", "pool": "D8/D12 + G4", "outcome": "Graph realised marginal", "existing": "Yes", "complete": "COMPLETE", "source": "selection_action_space_repair_dev300_v2", "action": "Consolidated"},
        {"experiment": "Set-level correspondence", "dataset": "Development100", "pool": "selected K=8 sets", "outcome": "U/RCC/CRA/BiAlign/BestAlign", "existing": "Yes", "complete": "COMPLETE N=100", "source": "rq2b_set_correspondence_dev100_v1", "action": "CIs verified/recomputed"},
        {"experiment": "Independent confirmation", "dataset": "Test200", "pool": "frozen arms", "outcome": "realised ΔU@8", "existing": "Yes", "complete": "COMPLETE N=200", "source": "confirmatory_test200_v1", "action": "Copied from frozen artifact"},
    ]

    provenance_files = [
        query_path, admin_path, corpus_path, corpus_map_path, dense_path, graph_path, registry_path,
        ROOT / "out/development300_m50_preflight_v1/formal_union_manifest.jsonl",
        ROOT / "out/development200_sbert_official_v1/no_recognition.jsonl",
        ROOT / "out/development300_sbert_official_additional200_v1/no_recognition.jsonl",
        *historical_variant_paths.values(),
        selector_dir / "dense_summary.csv", selector_dir / "graph_summary.csv",
        selector_dir / "per_query_decomposition.parquet", set_dir / "rq2b_c_per_query.csv",
        test_path, bm25_path,
        ROOT / "out/dev100_v2_versioned_runs/runs/bm25s.jsonl",
        ROOT / "out/development200_shortlist_v1/runs/bm25s.jsonl",
        ROOT / "out/official_round2_dev100/round2_gates.json",
        ROOT / "out/query_rewrite_graph_entry_dev100_v2/label_blind_analysis/rewrite_entry_report.json",
        ROOT / "out/learned_diffusion_dev100_v1/seed_only_cv_report.json",
    ]
    provenance_rows = [{"path": str(path.relative_to(ROOT)), "sha256": rq2_sha256(path)} for path in provenance_files]

    blocked_rows = [
        {"experiment": "BM25 B4 effectiveness", "current": "N=300 candidate run; 822/1304 backend-union pairs judged", "missing": "482 utility-v2 query–candidate labels", "definition": "Yes; requires exact-payload external-judging authorisation and existing stability gate."},
        {"experiment": "Recognition/static principal route", "current": "N=172 shared raw query IDs", "missing": "full Development300 strict G4 materialisation and provenance-controlled ranking", "definition": "Potentially; rerun existing method on the frozen 300, without relabelling it as maintained Graph4."},
        {"experiment": "Structured reformulation", "current": "N=100; 89 entered", "missing": "validated reformulations and strict graph outputs for additional 200", "definition": "Potentially; requires frozen generation protocol and new external generation authorisation."},
        {"experiment": "Corrected PPR node+hub", "current": "N=172 shared frozen-run queries", "missing": "128 Dev300 queries plus full strict-native provenance", "definition": "Yes if the historical configuration is rerun unchanged."},
        {"experiment": "Local PCST", "current": "N=172 shared frozen-run queries", "missing": "128 Dev300 queries plus full strict-native provenance", "definition": "Yes if the historical implementation/hyperparameters are rerun unchanged."},
        {"experiment": "Learned propagation", "current": "N=28 entered-query subset; NO_GO", "missing": "frozen deployment checkpoint and full-cohort applicable output", "definition": "No for a full comparable row from current artifacts; training/deployment work would be a new experiment."},
    ]

    lines = [
        "# RQ2 Complete Experimental Data",
        "",
        f"Run ID: `rq2-complete-experiment-data-v1`; generated UTC: `{datetime.now(timezone.utc).isoformat()}`.",
        "",
        "## 1. Reproducibility and Dataset Audit",
        "",
        f"- Development300: `N=300`; nested Development100=`{component_counts.get('nested_development100', 0)}`; seeded additional set=`{component_counts.get('seeded_additional200', 0)}`; single-need=`{need_counts.get('single_need', 0)}`; multi-need=`{need_counts.get('multi_need', 0)}`.",
        f"- Query-ID SHA-256: `7e2e131e8023d08718e21764eac056c6f0bb23e529085fe5b1bd5c5f989fe66e`; normalized-query file SHA-256: `{rq2_sha256(query_path)}`.",
        f"- Candidate corpus: `N={len(rq2_read_json(corpus_path))}` comments; SHA-256: `{rq2_sha256(corpus_path)}`.",
        f"- Utility registry: `N={len(utility)}` unique query–candidate pairs; utility-v2; SHA-256: `{rq2_sha256(registry_path)}`.",
        "- Final evidence budget: `K=8`. Dense depths: `8, 12, 20, 50`. Graph quota: `4`.",
        "- Development300 bootstrap: whole-query paired bootstrap; 5,000 draws; seed `20260805`; common-index SHA-256 `f1804695368b8bbea9d70bb9e39b4764ac22c4fdc552c40b2f7308d985f09b8a`.",
        "- Set-level Development100 bootstrap: whole-query paired bootstrap; 5,000 draws; seed `20260810`; saved-index SHA-256 `e32e355898aeb8e687857ce70ae845fa481f7ef55620cadcf9139e95fa2a34d3`.",
        "- Tie handling for selector outputs: prefer fewer replacements, then candidate ID. Oracle ties: utility descending, candidate ID for deterministic materialisation; the mean top-8 utility is tie-invariant.",
        "- Same-thread audit: formal Development300 candidate union, no-recognition G4, and newly materialised BM25 B4 each have zero source-post matches for both anchors.",
        "- Retrieval construction flags in frozen manifests: utility not used; community-response correspondence not used; Test200 not read during Development300 construction/training.",
        "- Maintained Graph4 identity: label-blind round-robin shortlist over strict `no_recognition` and `fact_only_no_recognition` PPR routes; 1,200/1,200 native rows; fallback/callback/padding = 0/0/0. It is not the historical recognition-entry static-PPR route named in the requested variant list.",
        "",
        "## 2. Experiment Coverage Matrix",
        "",
        rq2_markdown_table(coverage_matrix, [("experiment", "Experiment"), ("dataset", "Dataset"), ("pool", "Candidate pool"), ("outcome", "Outcome"), ("existing", "Existing?"), ("complete", "Complete?"), ("source", "Source file/script"), ("action", "Action")]),
        "",
        "## 3. Dense Candidate Availability — Development300",
        "",
        "### Table A — Dense candidate availability",
        "",
        rq2_markdown_table(a_rows, [("backend", "Backend"), ("pool", "Pool"), ("N", "N"), ("mean", "Mean pool U"), ("oracle", "Oracle U@8"), ("delta", "Δ Oracle vs D8"), ("ci", "95% CI"), ("useful", "Useful U≥4")]),
        "",
        "## 4. Principal Graph Candidate Frontier — Development300",
        "",
        "### Table B — Principal Graph candidate-access frontier",
        "",
        rq2_markdown_table(b_rows, [("backend", "Backend"), ("M", "M"), ("N", "N"), ("dense", "Dense Oracle U@8"), ("union", "Dense+Graph Oracle U@8"), ("delta", "Graph marginal"), ("ci", "95% CI"), ("unique", "Graph unique fraction"), ("added", "Mean added U"), ("count", "Mean added count")]),
        "",
        f"Observed marginal range: MiniLM `{rq2_fmt(next(x['marginal'] for x in table_b if x['backend']=='minilm' and x['M']==8))}` at M=8 to `{rq2_fmt(next(x['marginal'] for x in table_b if x['backend']=='minilm' and x['M']==50))}` at M=50; E5 `{rq2_fmt(next(x['marginal'] for x in table_b if x['backend']=='e5' and x['M']==8))}` to `{rq2_fmt(next(x['marginal'] for x in table_b if x['backend']=='e5' and x['M']==50))}`.",
        "",
        "## 5. Unified Graph Variant Comparison — Development300",
        "",
        "### Table C — Unified Development300 Graph-variant comparison",
        "",
        rq2_markdown_table(c_rows, [("backend", "Backend"), ("variant", "Graph variant"), ("status", "Status"), ("N", "Valid N"), ("count", "Mean added"), ("unique", "Unique vs D8"), ("added", "Mean added U"), ("oracle", "Oracle U@8"), ("delta", "Δ vs D8"), ("ci", "95% CI"), ("principal", "Δ vs principal"), ("principal_ci", "95% CI"), ("reason", "Audit note")]),
        "",
        "## 6. BM25 Shallow Control — Development300",
        "",
        "### Table D — BM25 versus Graph shallow access",
        "",
        rq2_markdown_table(d_rows, [("backend", "Backend"), ("route", "Added route"), ("status", "Status"), ("oracle", "Oracle U@8"), ("delta", "Δ vs D8"), ("ci", "95% CI"), ("unique", "Unique fraction"), ("added", "Mean added U"), ("graph", "Δ vs Graph4"), ("graph_ci", "95% CI")]),
        "",
        f"BM25 candidate retrieval is complete for 300 queries and reproduces prior Top-100 rankings exactly on 100/100 nested-Dev100 queries and 172/172 queries shared with the frozen Development200 run. Utility-complete B4 queries are MiniLM {bm25_complete_queries['minilm']}/300 and E5 {bm25_complete_queries['e5']}/300; these subsets were not used for effectiveness estimates.",
        "",
        "## 7. Matched-Budget Oracle Comparison — Development300",
        "",
        "### Table E — Development300 matched-budget Oracle comparison",
        "",
        rq2_markdown_table(e_rows, [("backend", "Backend"), ("M", "M"), ("dense_pool", "All-Dense pool"), ("matched_pool", "Matched Graph pool"), ("dense", "Dense Oracle"), ("matched", "Matched Oracle"), ("delta", "Δ matched−Dense"), ("ci", "95% CI"), ("unique", "Graph unique fraction"), ("count", "Final count")]),
        "",
        "## 8. Utility Conversion — Development300",
        "",
        "### Table F — Complete Development300 utility-conversion matrix (r=1,2,8)",
        "",
        rq2_markdown_table([row for row in f_rows if row["r"] in (1, 2, 8)], [("backend", "Backend"), ("M", "M"), ("scorer", "Scorer"), ("r", "r"), ("delta", "Realised ΔU@8"), ("ci", "95% CI"), ("absolute", "Absolute U@8"), ("oracle", "Oracle headroom"), ("conversion", "Conversion"), ("harm", "Harm rate"), ("wtl", "W/T/L")]),
        "",
        "### Table F-supplement — Completed r=4 cells",
        "",
        rq2_markdown_table([row for row in f_rows if row["r"] == 4], [("backend", "Backend"), ("M", "M"), ("scorer", "Scorer"), ("r", "r"), ("delta", "Realised ΔU@8"), ("ci", "95% CI"), ("absolute", "Absolute U@8"), ("oracle", "Oracle headroom"), ("conversion", "Conversion"), ("harm", "Harm rate"), ("wtl", "W/T/L")]),
        "",
        "### Table G — Oracle versus realised conversion",
        "",
        rq2_markdown_table([row for row in g_rows if row["r"] in (1, 2, 8)], [("backend", "Backend"), ("M", "M"), ("scorer", "Scorer"), ("r", "r"), ("raw", "Raw U@8"), ("oracle_abs", "Oracle U@8"), ("real_abs", "Realised U@8"), ("access", "Access Δ"), ("real", "Realised Δ"), ("gap", "Unconverted gap"), ("conversion", "Conversion")]),
        "",
        "## 9. Graph Utility Conversion — Development300",
        "",
        "### Table H — Graph availability-to-conversion comparison",
        "",
        rq2_markdown_table(h_rows, [("backend", "Backend"), ("M", "M"), ("scorer", "Scorer"), ("capacity", "Selection capacity"), ("oracle", "Oracle Graph marginal"), ("real", "Realised Graph marginal"), ("ci", "95% CI"), ("conversion", "Conversion"), ("precision", "Entrant precision"), ("harm", "Harm rate")]),
        "",
        "## 10. Selected-Set Community Correspondence",
        "",
        "### Table I — Selected-set Development100 evaluation",
        "",
        "Each metric cell is `absolute mean; Δ vs Raw [paired 95% CI]`. N=100.",
        "",
        rq2_markdown_table(i_rows, [("method", "Method"), ("N", "N"), ("u", "U@8"), ("rcc", "RCC"), ("cra", "CRA"), ("bi", "BiAlignF1"), ("best", "BestAlign")]),
        "",
        "MMR uses λ=0.1 and r=2. Its tie-break is quality, then D8 membership, then candidate ID; the saved audit marks this tie-break as not configured-identical to the additive selector.",
        "",
        "## 11. Independent Test200",
        "",
        "### Table J — Frozen Test200 contrasts",
        "",
        rq2_markdown_table(j_rows, [("backend", "Backend"), ("contrast", "Contrast"), ("N", "N"), ("delta", "ΔU@8"), ("ci", "95% CI"), ("wtl", "W/T/L")]),
        "",
        "## 12. Utility Coverage Audit",
        "",
        "### Table K — Utility coverage audit",
        "",
        rq2_markdown_table(k_rows, [("variant", "Variant"), ("pairs", "Unique query-candidate pairs"), ("already", "Already judged"), ("new", "Newly judged"), ("missing", "Missing"), ("coverage", "Coverage")]),
        "",
        "New external utility judgments in this consolidation run: `0`. Missing labels were not imputed. Any further provider call requires new exact-payload authorisation under the existing project gate.",
        "",
        "## 13. Graph Intermediate Diagnostics",
        "",
        rq2_markdown_table(diag_rows, [("dataset", "Dataset"), ("method", "Method"), ("scope", "Scope"), ("metric", "Metric"), ("value", "Observed value"), ("detail", "Additional data")]),
        "",
        "- Local PCST historical execution scope: Development100 candidate-distribution analysis; graph entry on 29% in the earlier scope; mean selected nodes 44.24, mean selected edges 43.24, mean runtime 0.336 s/query, mean PCST-derived passages 140.86. Source: `docs_v2/45_Official_HippoRAG2_PCST与传播变体_2026-07-17.md` and frozen transition artifacts.",
        "- Maintained fixed Graph4 provenance: both-route candidates 1,112; no-recognition-only 67; fact-only-no-recognition-only 21; valid pre-fallback route memberships 2,312; pre-fallback rank range 1–96; mean candidate utility 3.7393. Scope: Development300, 1,200 candidates.",
        "- Learned propagation frozen entered-query values (N=28): seed+head vs static PPR judged-only nDCG@3 Δ=0.0686, 95% CI [0.0045, 0.1294]; mean Utility@3 Δ=0.2339, 95% CI [0.0042, 0.4737]; pairwise accuracy Δ=0.1383, 95% CI [0.1065, 0.1713]; Spearman Δ=0.0302, 95% CI [-0.0143, 0.0774]. Final decision: `NO_GO_LEARNED_EDGE_DIFFUSION`.",
        "",
        "## 14. Missing or Blocked Experiments",
        "",
        rq2_markdown_table(blocked_rows, [("experiment", "Experiment"), ("current", "Current scope"), ("missing", "Exact missing item"), ("definition", "Completable without definition change?")]),
        "",
        "## 15. Artifact Provenance",
        "",
        "### Source hashes",
        "",
        rq2_markdown_table(provenance_rows, [("path", "Source path"), ("sha256", "SHA-256")]),
        "",
        "### Table-to-artifact map",
        "",
        rq2_markdown_table([
            {"table": "A", "source": "evidence_selection/run_selection_action_space_repair.py; dense_m50_memberships.jsonl; complete utility registry", "output": "out/rq2_complete_experiment_data_v1/table_a_dense_candidate_availability.csv", "status": "Verified/recomputed; 20260805/5000"},
            {"table": "B", "source": "tools/prepare_development300_m50_preflight.py; strict_fixed_graph4.jsonl", "output": "out/rq2_complete_experiment_data_v1/table_b_principal_graph_frontier.csv", "status": "Verified/recomputed; 20260805/5000"},
            {"table": "C", "source": "historical Graph run inventory + Table B + registry coverage", "output": "out/rq2_complete_experiment_data_v1/table_c_graph_variant_comparison.csv", "status": "Mixed COMPLETE/PARTIAL/BLOCKED; no cross-scope pooling"},
            {"table": "D", "source": "tools/run_section17_pipeline.py bm25s; normalized BM25 run", "output": "out/rq2_complete_experiment_data_v1/table_d_bm25_vs_graph.csv", "status": "Candidates complete; effectiveness blocked"},
            {"table": "E", "source": "existing rule in tools/run_m50_matched_budget_analysis.py; frozen Dev300 inputs", "output": "out/rq2_complete_experiment_data_v1/table_e_matched_budget_oracle.csv", "status": "Newly recomputed; 20260805/5000"},
            {"table": "F/G/H", "source": "experiments/selection_action_space_repair_dev300_v2/*.csv", "output": "out/rq2_complete_experiment_data_v1/table_f_utility_conversion.csv; table_g_oracle_realised_conversion.csv; table_h_graph_conversion.csv", "status": "Existing results consolidated"},
            {"table": "I", "source": "experiments/rq2b_set_correspondence_dev100_v1/rq2b_c_per_query.csv", "output": "out/rq2_complete_experiment_data_v1/table_i_selected_set_dev100.csv", "status": "Existing means; all method-vs-Raw CIs recomputed 20260810/5000"},
            {"table": "J", "source": "out/confirmatory_test200_v1/analysis_v1_output_union/paired_contrasts.json", "output": "out/rq2_complete_experiment_data_v1/table_j_independent_test200.csv", "status": "Frozen result copied exactly"},
            {"table": "K", "source": "variant candidate IDs joined to frozen complete utility registry", "output": "out/rq2_complete_experiment_data_v1/table_k_utility_coverage.csv", "status": "New local audit; external calls=0"},
        ], [("table", "Table"), ("source", "Source scripts/files"), ("output", "Generated intermediate"), ("status", "Run/seed/config")]),
        "",
        "Generated by `evaluation/consolidate_rq2_complete_experiment_data.py`. No `.tex` files were read for values or modified by this run.",
        "The preliminary files `out/rq2_complete_experiment_data_v1/bm25_development300.jsonl` and its manifest are superseded because their query-ID namespace was not normalized. All tables use `bm25_development300_normalized.jsonl` only.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    outputs = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "reproduction_manifest.json")
    manifest = {
        "version": "rq2-complete-experiment-data-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "external_calls": 0,
        "thesis_tex_modified": False,
        "development_queries": 300,
        "final_k": FINAL_K,
        "bootstrap": {"samples": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "whole query"},
        "same_thread_bm25_leaks": {backend: len(rows) for backend, rows in same_thread_leaks.items()},
        "same_thread_no_recognition_leaks": {backend: len(rows) for backend, rows in no_rec_same_thread_leaks.items()},
        "bm25_reproduction": bm25_reproduction,
        "bm25_utility_complete_queries": bm25_complete_queries,
        "source_hashes": {str(path.relative_to(ROOT)): rq2_sha256(path) for path in provenance_files},
        "output_hashes": {str(path.relative_to(ROOT)): rq2_sha256(path) for path in outputs},
        "report_path": str(REPORT.relative_to(ROOT)),
        "report_sha256": rq2_sha256(REPORT),
        "implementation": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": rq2_sha256(Path(__file__).resolve()),
        },
    }
    (OUT / "reproduction_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
