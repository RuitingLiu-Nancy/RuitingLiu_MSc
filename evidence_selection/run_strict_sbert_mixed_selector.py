#!/usr/bin/env python3
"""Nested-OOF mixed-pool selectors for the fully judged strict-SBERT pool.

This experiment answers a narrower question than the Oracle analysis:
can a deployable selector recover graph candidates that add utility beyond
the same-backend Dense depth control?

The implementation deliberately keeps four boundaries explicit:

* development-only, with frozen test paths rejected;
* one exact D8+Depth4+Graph4 pool, fully judged before training;
* query-grouped outer 5x5 OOF evaluation;
* matched graph/no-graph contrasts that keep fitted models fixed.

Canonical pairwise Linear, XGBoost LambdaMART, feature scaling, graded nDCG,
and query bootstrap implementations are reused from the project.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# macOS native runtimes in this repository require XGBoost to be loaded before
# the canonical reranker module imports Torch.  Keep this ordering explicit;
# the model is still constructed by canonical.fit_lambdamart.
import xgboost as _xgboost  # noqa: F401

try:
    import configuration as project_config
    from evaluation.ir_metrics import graded_ndcg_at
    from evaluation.judgment_completeness import (
        complete_utility_v2_rows,
    )
    from evaluation.statistics import bootstrap_ci
    from utility_scoring.learned_diffusion import reranker_validation as canonical
    from candidate_pool.analyze_strict_sbert_graph_oracle import (
        _read_jsonl,
        _reject_test,
        _sha256,
        _write_json,
        _write_jsonl,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import configuration as project_config
    from evaluation.ir_metrics import graded_ndcg_at
    from evaluation.judgment_completeness import (
        complete_utility_v2_rows,
    )
    from evaluation.statistics import bootstrap_ci
    from utility_scoring.learned_diffusion import reranker_validation as canonical
    from candidate_pool.analyze_strict_sbert_graph_oracle import (
        _read_jsonl,
        _reject_test,
        _sha256,
        _write_json,
        _write_jsonl,
    )


TOKEN = re.compile(r"(?u)\b\w\w+\b")
DENSE_TOP = "dense_top8"
DENSE_DEPTH = "dense_depth_control"
GRAPH = "graph_beyond_dense100_route_balanced"
BASIC_FEATURES = (
    "dense_score",
    "dense_rank_reciprocal",
    "dense_missing",
    "source_rank_reciprocal",
    "comment_length_log",
    "query_length_log",
    "lexical_jaccard",
    "lexical_query_coverage",
)
ROUTE_FEATURES = (
    "source_is_dense_top8",
    "source_is_dense_depth",
    "source_is_graph",
    "no_recognition_score",
    "no_recognition_rank_reciprocal",
    "no_recognition_missing",
    "fact_only_no_recognition_score",
    "fact_only_no_recognition_rank_reciprocal",
    "fact_only_no_recognition_missing",
    "graph_route_count",
    "graph_route_consensus",
    "graph_best_rank_reciprocal",
    "graph_mean_score",
    "outside_dense100",
)


def _selector_tokens(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))


def _query_minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high <= low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _run_features(path: Path, eligible_qids: set[str]) -> dict[tuple[str, str], dict]:
    """Load rank and query-local normalized score from an Official run."""
    _reject_test(path)
    result: dict[tuple[str, str], dict] = {}
    for row in _read_jsonl(path):
        qid = str(row["query_id"])
        if qid not in eligible_qids:
            continue
        ids = [str(value) for value in row["retrieved_titles"]]
        scores = [float(value) for value in row["retrieved_scores"]]
        if len(ids) != len(scores):
            raise ValueError(f"{path}: id/score length mismatch for {qid}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path}: duplicate candidate for {qid}")
        for rank, (cid, score) in enumerate(
                zip(ids, _query_minmax(scores), strict=True), start=1):
            result[(qid, cid)] = {"rank": rank, "score": score}
    return result


@dataclass
class MixedDataset:
    """Dataset interface consumed by the canonical reranker trainers."""

    rows: list[dict]
    qrels: dict[tuple[str, str], float]

    def __post_init__(self) -> None:
        self.by_query: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            self.by_query[str(row["query_id"])].append(index)
        self.basic = np.asarray([
            [float(row["basic_features"][name]) for name in BASIC_FEATURES]
            for row in self.rows
        ], dtype=np.float32)
        self.graph = np.asarray([
            [float(row["route_features"][name]) for name in ROUTE_FEATURES]
            for row in self.rows
        ], dtype=np.float32)
        self.utility = np.asarray([
            self.qrels.get((str(row["query_id"]), str(row["comment_id"])), np.nan)
            for row in self.rows
        ], dtype=np.float32)


@dataclass
class PairDataset:
    """Two-item pseudo-query groups for graph-vs-dense-tail supervision."""

    rows: list[dict]
    basic: np.ndarray
    graph: np.ndarray
    utility: np.ndarray

    def __post_init__(self) -> None:
        self.by_query: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            self.by_query[str(row["query_id"])].append(index)


def _build_mixed_feature_rows(
    pool_rows: list[dict],
    dense: dict[tuple[str, str], dict],
    graph_runs: dict[str, dict[tuple[str, str], dict]],
) -> list[dict]:
    output = []
    for row in pool_rows:
        qid = str(row["query_id"])
        cid = str(row["comment_id"])
        source = str(row["candidate_source"])
        query_text = str(row.get("query_text") or "")
        comment_text = str(row.get("comment_text") or "")
        q_tokens = _selector_tokens(query_text)
        c_tokens = _selector_tokens(comment_text)
        intersection = len(q_tokens & c_tokens)
        union = len(q_tokens | c_tokens)
        dense_value = dense.get((qid, cid))

        route_values = {
            name: values.get((qid, cid))
            for name, values in graph_runs.items()
        }
        present = [value for value in route_values.values() if value is not None]

        def route_triplet(name: str) -> tuple[float, float, float]:
            value = route_values[name]
            if value is None:
                return 0.0, 0.0, 1.0
            return (
                float(value["score"]),
                1.0 / (60.0 + float(value["rank"])),
                0.0,
            )

        no_score, no_rank, no_missing = route_triplet("no_recognition")
        fact_score, fact_rank, fact_missing = route_triplet(
            "fact_only_no_recognition")
        best_rank = max(
            (1.0 / (60.0 + float(value["rank"])) for value in present),
            default=0.0,
        )
        mean_score = (
            statistics.fmean(float(value["score"]) for value in present)
            if present else 0.0
        )
        output.append({
            **row,
            "query_id": qid,
            "comment_id": cid,
            "basic_features": dict(zip(BASIC_FEATURES, (
                float(dense_value["score"]) if dense_value else 0.0,
                1.0 / (60.0 + float(dense_value["rank"]))
                if dense_value else 0.0,
                float(dense_value is None),
                1.0 / (60.0 + float(row["source_rank"])),
                math.log1p(len(comment_text)),
                math.log1p(len(query_text)),
                intersection / max(1, union),
                intersection / max(1, len(q_tokens)),
            ), strict=True)),
            "route_features": dict(zip(ROUTE_FEATURES, (
                float(source == DENSE_TOP),
                float(source == DENSE_DEPTH),
                float(source == GRAPH),
                no_score,
                no_rank,
                no_missing,
                fact_score,
                fact_rank,
                fact_missing,
                float(len(present)),
                float(len(present) > 1),
                best_rank,
                mean_score,
                float(dense_value is None),
            ), strict=True)),
            "feature_construction_used_utility": False,
        })
    return output


def _validate_inputs(
    dataset: MixedDataset,
    split_manifest: dict,
    *,
    top_k: int,
) -> dict:
    qids = set(dataset.by_query)
    sources = Counter(str(row["candidate_source"]) for row in dataset.rows)
    expected_sources = {
        DENSE_TOP: len(qids) * top_k,
        DENSE_DEPTH: len(qids) * 4,
        GRAPH: len(qids) * 4,
    }
    if sources != expected_sources:
        raise ValueError(
            f"candidate source contract mismatch: {sources} != {expected_sources}")
    if len(dataset.rows) != len({
        (row["query_id"], row["comment_id"]) for row in dataset.rows
    }):
        raise ValueError("candidate pool contains duplicate query-comment pairs")
    missing = int(np.sum(~np.isfinite(dataset.utility)))
    if missing:
        raise ValueError(f"candidate pool has {missing} missing utility labels")
    if any(bool(row.get("selection_used_utility")) for row in dataset.rows):
        raise ValueError("candidate pool selection read utility")

    validation_counts: dict[tuple[int, str], int] = Counter()
    filtered_fold_sizes = []
    for fold_row in split_manifest["rows"]:
        train = set(map(str, fold_row["train_query_ids"])) & qids
        valid = set(map(str, fold_row["validation_query_ids"])) & qids
        if train & valid:
            raise ValueError("query overlap within outer fold")
        if train | valid != qids:
            raise ValueError("filtered outer fold does not partition strict pool")
        filtered_fold_sizes.append({
            "repeat": int(fold_row["repeat"]),
            "fold": int(fold_row["fold"]),
            "train_queries": len(train),
            "validation_queries": len(valid),
        })
        for qid in valid:
            validation_counts[(int(fold_row["repeat"]), qid)] += 1
    repeats = sorted({int(row["repeat"]) for row in split_manifest["rows"]})
    if any(validation_counts[(repeat, qid)] != 1
           for repeat in repeats for qid in qids):
        raise ValueError("each query must appear once in validation per repeat")
    return {
        "queries": len(qids),
        "candidate_pairs": len(dataset.rows),
        "source_counts": dict(sources),
        "missing_utility_pairs": missing,
        "repeats": len(repeats),
        "folds_per_repeat": dict(Counter(
            int(row["repeat"]) for row in split_manifest["rows"])),
        "validation_appearances_per_query": len(repeats),
        "query_overlap_per_fold": 0,
        "filtered_fold_sizes": filtered_fold_sizes,
        "test_read": False,
    }


def _candidate_indices(
    dataset: MixedDataset, qid: str, *, include_graph: bool,
) -> list[int]:
    return [
        index for index in dataset.by_query[qid]
        if include_graph
        or dataset.rows[index]["candidate_source"] != GRAPH
    ]


def _rank(
    dataset: MixedDataset,
    qid: str,
    scores: np.ndarray,
    *,
    include_graph: bool,
) -> list[int]:
    return sorted(
        _candidate_indices(dataset, qid, include_graph=include_graph),
        key=lambda index: (
            -float(scores[index]),
            str(dataset.rows[index]["comment_id"]),
        ),
    )


def _raw_dense_order(dataset: MixedDataset, qid: str) -> list[int]:
    return sorted(
        (
            index for index in dataset.by_query[qid]
            if dataset.rows[index]["candidate_source"] == DENSE_TOP
        ),
        key=lambda index: (
            int(dataset.rows[index]["source_rank"]),
            str(dataset.rows[index]["comment_id"]),
        ),
    )


def _pair_dataset(
    dataset: MixedDataset,
    original_qids: list[str],
    *,
    dense_tail_start_rank: int,
    margin: float,
) -> PairDataset:
    rows: list[dict] = []
    basics: list[np.ndarray] = []
    routes: list[np.ndarray] = []
    utilities: list[float] = []
    for qid in original_qids:
        graph_indices = [
            index for index in dataset.by_query[qid]
            if dataset.rows[index]["candidate_source"] == GRAPH
        ]
        tail_indices = [
            index for index in dataset.by_query[qid]
            if dataset.rows[index]["candidate_source"] == DENSE_DEPTH
            or (
                dataset.rows[index]["candidate_source"] == DENSE_TOP
                and int(dataset.rows[index]["source_rank"]) >= dense_tail_start_rank
            )
        ]
        for graph_index in graph_indices:
            for dense_index in tail_indices:
                difference = (
                    float(dataset.utility[graph_index])
                    - float(dataset.utility[dense_index])
                )
                if abs(difference) < margin:
                    continue
                pseudo_qid = (
                    f"{qid}::g={dataset.rows[graph_index]['comment_id']}"
                    f"::d={dataset.rows[dense_index]['comment_id']}"
                )
                for index in (graph_index, dense_index):
                    rows.append({
                        "query_id": pseudo_qid,
                        "comment_id": dataset.rows[index]["comment_id"],
                        "original_query_id": qid,
                        "candidate_source": dataset.rows[index]["candidate_source"],
                    })
                    basics.append(dataset.basic[index])
                    routes.append(dataset.graph[index])
                    utilities.append(float(dataset.utility[index]))
    if not rows:
        raise ValueError("no graph-vs-dense-tail training pairs")
    return PairDataset(
        rows=rows,
        basic=np.asarray(basics, dtype=np.float32),
        graph=np.asarray(routes, dtype=np.float32),
        utility=np.asarray(utilities, dtype=np.float32),
    )


def _marginal_accuracy(
    dataset: MixedDataset,
    scores: np.ndarray,
    qids: list[str],
    *,
    dense_tail_start_rank: int,
    margin: float,
) -> float:
    per_query = []
    for qid in qids:
        graph_indices = [
            index for index in dataset.by_query[qid]
            if dataset.rows[index]["candidate_source"] == GRAPH
        ]
        tail_indices = [
            index for index in dataset.by_query[qid]
            if dataset.rows[index]["candidate_source"] == DENSE_DEPTH
            or (
                dataset.rows[index]["candidate_source"] == DENSE_TOP
                and int(dataset.rows[index]["source_rank"]) >= dense_tail_start_rank
            )
        ]
        comparisons = []
        for graph_index in graph_indices:
            for dense_index in tail_indices:
                difference = (
                    float(dataset.utility[graph_index])
                    - float(dataset.utility[dense_index])
                )
                if abs(difference) < margin:
                    continue
                predicted = float(scores[graph_index]) - float(scores[dense_index])
                comparisons.append((predicted > 0) == (difference > 0))
        if comparisons:
            per_query.append(float(np.mean(comparisons)))
    return statistics.fmean(per_query) if per_query else float("nan")


def _marginal_label_distribution(
    dataset: MixedDataset,
    qids: list[str],
    *,
    dense_tail_start_rank: int,
    margin: float,
) -> dict:
    """Report the directional class imbalance behind pairwise accuracy."""
    graph_wins = 0
    dense_wins = 0
    omitted_within_margin = 0
    per_query_graph_win_rate = []
    for qid in qids:
        graph_indices = [
            index for index in dataset.by_query[qid]
            if dataset.rows[index]["candidate_source"] == GRAPH
        ]
        tail_indices = [
            index for index in dataset.by_query[qid]
            if dataset.rows[index]["candidate_source"] == DENSE_DEPTH
            or (
                dataset.rows[index]["candidate_source"] == DENSE_TOP
                and int(dataset.rows[index]["source_rank"]) >= dense_tail_start_rank
            )
        ]
        query_graph_wins = 0
        query_dense_wins = 0
        for graph_index in graph_indices:
            for dense_index in tail_indices:
                difference = (
                    float(dataset.utility[graph_index])
                    - float(dataset.utility[dense_index])
                )
                if difference >= margin:
                    graph_wins += 1
                    query_graph_wins += 1
                elif difference <= -margin:
                    dense_wins += 1
                    query_dense_wins += 1
                else:
                    omitted_within_margin += 1
        eligible = query_graph_wins + query_dense_wins
        if eligible:
            per_query_graph_win_rate.append(query_graph_wins / eligible)
    eligible = graph_wins + dense_wins
    micro_graph_win_rate = graph_wins / eligible
    macro_graph_win_rate = statistics.fmean(per_query_graph_win_rate)
    return {
        "eligible_directional_pairs": eligible,
        "graph_wins": graph_wins,
        "dense_tail_wins": dense_wins,
        "omitted_within_margin": omitted_within_margin,
        "micro_graph_win_rate": micro_graph_win_rate,
        "micro_always_dense_accuracy": 1.0 - micro_graph_win_rate,
        "query_macro_graph_win_rate": macro_graph_win_rate,
        "query_macro_always_dense_accuracy": 1.0 - macro_graph_win_rate,
    }


def _fit_marginal(
    dataset: MixedDataset,
    train_qids: list[str],
    train_cfg: dict,
    *,
    dense_tail_start_rank: int,
    seed: int,
) -> tuple[object, object, np.ndarray, float, list[dict]]:
    """Tune and fit pairwise graph-vs-dense-tail Linear inside outer train."""
    l2_values = list(train_cfg["linear_l2"])
    margin = float(train_cfg["pair_margin"])
    traces = []
    folds = canonical.inner_folds(
        train_qids, int(train_cfg["inner_folds"]), seed)
    for config_index, l2 in enumerate(l2_values):
        fold_scores = []
        for inner_index, (inner_train, inner_valid) in enumerate(folds):
            scaler = canonical.fit_scaler(dataset, inner_train, "graph")
            full_matrix = canonical.matrix(dataset, scaler, "graph")
            pseudo = _pair_dataset(
                dataset,
                inner_train,
                dense_tail_start_rank=dense_tail_start_rank,
                margin=margin,
            )
            pseudo_matrix = scaler.transform(
                np.hstack((pseudo.basic, pseudo.graph))).astype(np.float32)
            model = canonical.fit_linear(
                pseudo,
                pseudo_matrix,
                sorted(pseudo.by_query),
                float(l2),
                train_cfg,
                seed + config_index * 97 + inner_index,
            )
            scores = canonical.predict_model("linear", model, full_matrix)
            fold_scores.append(_marginal_accuracy(
                dataset,
                scores,
                inner_valid,
                dense_tail_start_rank=dense_tail_start_rank,
                margin=margin,
            ))
        traces.append({
            "l2": float(l2),
            "inner_marginal_pairwise_accuracy": statistics.fmean(fold_scores),
            "fold_scores": fold_scores,
        })
    best_index = max(
        range(len(traces)),
        key=lambda index: (
            traces[index]["inner_marginal_pairwise_accuracy"],
            -index,
        ),
    )
    best_l2 = float(l2_values[best_index])
    scaler = canonical.fit_scaler(dataset, train_qids, "graph")
    full_matrix = canonical.matrix(dataset, scaler, "graph")
    pseudo = _pair_dataset(
        dataset,
        train_qids,
        dense_tail_start_rank=dense_tail_start_rank,
        margin=margin,
    )
    pseudo_matrix = scaler.transform(
        np.hstack((pseudo.basic, pseudo.graph))).astype(np.float32)
    model = canonical.fit_linear(
        pseudo,
        pseudo_matrix,
        sorted(pseudo.by_query),
        best_l2,
        train_cfg,
        seed,
    )
    scores = canonical.predict_model("linear", model, full_matrix)
    return model, scaler, scores, best_l2, traces


def _fit_pointwise(
    dataset: MixedDataset,
    train_qids: list[str],
    train_cfg: dict,
    *,
    family: str,
    feature_set: str,
    seed: int,
) -> tuple[np.ndarray, object, list[dict]]:
    configs = (
        list(train_cfg["linear_l2"])
        if family == "linear"
        else list(train_cfg["lambdamart_configs"])
    )
    selected, trace = canonical.tune(
        dataset,
        train_qids,
        family,
        feature_set,
        configs,
        train_cfg,
        seed,
    )
    scaler = canonical.fit_scaler(dataset, train_qids, feature_set)
    full_matrix = canonical.matrix(dataset, scaler, feature_set)
    if family == "linear":
        model = canonical.fit_linear(
            dataset,
            full_matrix,
            train_qids,
            float(selected),
            train_cfg,
            seed,
        )
    else:
        model = canonical.fit_lambdamart(
            dataset,
            full_matrix,
            train_qids,
            selected,
            seed,
        )
    return (
        canonical.predict_model(family, model, full_matrix),
        selected,
        trace,
    )


def _scale_subset(scores: np.ndarray, indices: list[int]) -> dict[int, float]:
    values = [float(scores[index]) for index in indices]
    normalized = _query_minmax(values)
    return dict(zip(indices, normalized, strict=True))


def _text_similarity(left: str, right: str) -> float:
    left_tokens, right_tokens = (
        _selector_tokens(left), _selector_tokens(right))
    return len(left_tokens & right_tokens) / max(
        1, len(left_tokens | right_tokens))


def _sequential_select(
    dataset: MixedDataset,
    qid: str,
    base_scores: np.ndarray,
    marginal_scores: np.ndarray,
    *,
    include_graph: bool,
    top_k: int,
    protected_dense: int,
    redundancy_weight: float,
    marginal_weight: float,
) -> list[int]:
    """Label-blind greedy Top-k selection with an optional graph marginal term."""
    indices = _candidate_indices(
        dataset, qid, include_graph=include_graph)
    base = _scale_subset(base_scores, indices)
    marginal = _scale_subset(marginal_scores, indices)
    protected = sorted(
        (
            index for index in indices
            if dataset.rows[index]["candidate_source"] == DENSE_TOP
            and int(dataset.rows[index]["source_rank"]) <= protected_dense
        ),
        key=lambda index: int(dataset.rows[index]["source_rank"]),
    )[:top_k]
    selected = list(protected)
    remaining = set(indices) - set(selected)
    while remaining and len(selected) < top_k:
        def objective(index: int) -> tuple[float, float, str]:
            row = dataset.rows[index]
            graph_term = 0.0
            if row["candidate_source"] == GRAPH:
                graph_term = marginal_weight * (marginal[index] - 0.5)
            redundancy = max(
                (
                    _text_similarity(
                        str(row.get("comment_text") or ""),
                        str(dataset.rows[prior].get("comment_text") or ""),
                    )
                    for prior in selected
                ),
                default=0.0,
            )
            value = base[index] + graph_term - redundancy_weight * redundancy
            return (
                value,
                base[index],
                str(row["comment_id"]),
            )

        chosen = max(remaining, key=objective)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _query_metrics(
    dataset: MixedDataset,
    qid: str,
    selected: list[int],
    *,
    arm: str,
    repeat: int,
    fold: int,
    include_graph: bool,
    top_k: int,
) -> tuple[dict, dict]:
    if len(selected) != top_k:
        raise ValueError(f"{qid}/{arm}: expected Top-{top_k}, got {len(selected)}")
    ids = [str(dataset.rows[index]["comment_id"]) for index in selected]
    gains = {
        str(dataset.rows[index]["comment_id"]): float(dataset.utility[index])
        for index in dataset.by_query[qid]
    }
    utilities = [float(dataset.utility[index]) for index in selected]
    depth_indices = _candidate_indices(
        dataset, qid, include_graph=False)
    depth_utilities = sorted(
        (float(dataset.utility[index]) for index in depth_indices),
        reverse=True,
    )
    depth_threshold = depth_utilities[top_k - 1]
    opportunities = {
        index for index in dataset.by_query[qid]
        if dataset.rows[index]["candidate_source"] == GRAPH
        and float(dataset.utility[index]) > depth_threshold
    }
    selected_graph = {
        index for index in selected
        if dataset.rows[index]["candidate_source"] == GRAPH
    }
    captured = selected_graph & opportunities
    oracle_full = statistics.fmean(sorted(gains.values(), reverse=True)[:top_k])
    row = {
        "query_id": qid,
        "arm": arm,
        "repeat": repeat,
        "fold": fold,
        "include_graph": include_graph,
        "mean_utility_at8": statistics.fmean(utilities),
        "ndcg_at8": graded_ndcg_at(ids, gains, top_k),
        "regret_to_full_oracle_at8": (
            oracle_full - statistics.fmean(utilities)),
        "selected_graph_count": len(selected_graph),
        "strict_graph_opportunity_count": len(opportunities),
        "strict_graph_opportunity_captured": len(captured),
        "graph_opportunity_hit": (
            float(bool(captured)) if opportunities else None),
        "graph_opportunity_recall": (
            len(captured) / len(opportunities) if opportunities else None),
        "graph_entrant_precision": (
            len(captured) / len(selected_graph) if selected_graph else None),
        "selection_used_utility": False,
    }
    prediction = {
        **row,
        "ranked_comment_ids": ids,
        "candidate_sources": [
            str(dataset.rows[index]["candidate_source"]) for index in selected
        ],
        "utilities_for_evaluation_only": utilities,
    }
    return row, prediction


def _selection_grid(raw: dict) -> list[dict]:
    result = []
    for protected in raw["protected_dense_options"]:
        for redundancy in raw["redundancy_weights"]:
            for marginal in raw["marginal_weights"]:
                result.append({
                    "protected_dense": int(protected),
                    "redundancy_weight": float(redundancy),
                    "marginal_weight": float(marginal),
                })
    return result


def _tune_set_selector(
    dataset: MixedDataset,
    train_qids: list[str],
    base_scores: np.ndarray,
    marginal_scores: np.ndarray,
    grid: list[dict],
    *,
    top_k: int,
) -> tuple[dict, list[dict]]:
    """Choose set parameters on outer-train queries only."""
    traces = []
    for setting in grid:
        utilities, ndcgs = [], []
        for qid in train_qids:
            selected = _sequential_select(
                dataset,
                qid,
                base_scores,
                marginal_scores,
                include_graph=True,
                top_k=top_k,
                **setting,
            )
            ids = [str(dataset.rows[index]["comment_id"]) for index in selected]
            gains = {
                str(dataset.rows[index]["comment_id"]): float(
                    dataset.utility[index])
                for index in dataset.by_query[qid]
            }
            utilities.append(statistics.fmean(
                float(dataset.utility[index]) for index in selected))
            ndcgs.append(graded_ndcg_at(ids, gains, top_k))
        traces.append({
            "config": setting,
            "train_mean_utility_at8": statistics.fmean(utilities),
            "train_ndcg_at8": statistics.fmean(ndcgs),
        })
    best_index = max(
        range(len(traces)),
        key=lambda index: (
            traces[index]["train_mean_utility_at8"],
            traces[index]["train_ndcg_at8"],
            -traces[index]["config"]["protected_dense"],
            -traces[index]["config"]["redundancy_weight"],
            -traces[index]["config"]["marginal_weight"],
        ),
    )
    return dict(traces[best_index]["config"]), traces


def _fold_predictions(
    dataset: MixedDataset,
    fold_row: dict,
    train_cfg: dict,
    selection_cfg: dict,
    *,
    dense_tail_start_rank: int,
    top_k: int,
) -> tuple[list[dict], list[dict], dict]:
    qids = set(dataset.by_query)
    train_qids = sorted(set(map(str, fold_row["train_query_ids"])) & qids)
    valid_qids = sorted(
        set(map(str, fold_row["validation_query_ids"])) & qids)
    repeat = int(fold_row["repeat"])
    fold = int(fold_row["fold"])
    seed = int(fold_row["seed"])

    basic_linear, basic_linear_cfg, basic_linear_trace = _fit_pointwise(
        dataset, train_qids, train_cfg,
        family="linear", feature_set="basic", seed=seed)
    mixed_linear, mixed_linear_cfg, mixed_linear_trace = _fit_pointwise(
        dataset, train_qids, train_cfg,
        family="linear", feature_set="graph", seed=seed + 1000)
    mixed_lm, mixed_lm_cfg, mixed_lm_trace = _fit_pointwise(
        dataset, train_qids, train_cfg,
        family="lambdamart", feature_set="graph", seed=seed + 2000)
    _, _, marginal, marginal_l2, marginal_trace = _fit_marginal(
        dataset,
        train_qids,
        train_cfg,
        dense_tail_start_rank=dense_tail_start_rank,
        seed=seed + 3000,
    )
    grid = _selection_grid(selection_cfg)
    linear_set_cfg, linear_set_trace = _tune_set_selector(
        dataset,
        train_qids,
        mixed_linear,
        marginal,
        grid,
        top_k=top_k,
    )
    lm_set_cfg, lm_set_trace = _tune_set_selector(
        dataset,
        train_qids,
        mixed_lm,
        marginal,
        grid,
        top_k=top_k,
    )

    metric_rows, prediction_rows = [], []
    for qid in valid_qids:
        arm_orders = {
            "dense_top8_raw": _raw_dense_order(dataset, qid),
        }
        for name, scores in (
            ("basic_linear", basic_linear),
            ("mixed_linear", mixed_linear),
            ("mixed_lambdamart", mixed_lm),
        ):
            arm_orders[f"{name}_depth"] = _rank(
                dataset, qid, scores, include_graph=False)[:top_k]
            arm_orders[f"{name}_full"] = _rank(
                dataset, qid, scores, include_graph=True)[:top_k]
        for name, scores, setting in (
            ("sequential_mixed_linear", mixed_linear, linear_set_cfg),
            ("sequential_mixed_lambdamart", mixed_lm, lm_set_cfg),
        ):
            arm_orders[f"{name}_depth"] = _sequential_select(
                dataset,
                qid,
                scores,
                marginal,
                include_graph=False,
                top_k=top_k,
                **setting,
            )
            arm_orders[f"{name}_full"] = _sequential_select(
                dataset,
                qid,
                scores,
                marginal,
                include_graph=True,
                top_k=top_k,
                **setting,
            )
        for arm, selected in arm_orders.items():
            metric, prediction = _query_metrics(
                dataset,
                qid,
                selected,
                arm=arm,
                repeat=repeat,
                fold=fold,
                include_graph=arm.endswith("_full"),
                top_k=top_k,
            )
            metric_rows.append(metric)
            prediction_rows.append(prediction)
    fold_audit = {
        "repeat": repeat,
        "fold": fold,
        "seed": seed,
        "train_queries": len(train_qids),
        "validation_queries": len(valid_qids),
        "basic_linear_config": basic_linear_cfg,
        "mixed_linear_config": mixed_linear_cfg,
        "mixed_lambdamart_config": mixed_lm_cfg,
        "marginal_linear_l2": marginal_l2,
        "linear_set_config": linear_set_cfg,
        "lambdamart_set_config": lm_set_cfg,
        "validation_marginal_pairwise_accuracy": _marginal_accuracy(
            dataset,
            marginal,
            valid_qids,
            dense_tail_start_rank=dense_tail_start_rank,
            margin=float(train_cfg["pair_margin"]),
        ),
        "tuning": {
            "basic_linear": basic_linear_trace,
            "mixed_linear": mixed_linear_trace,
            "mixed_lambdamart": mixed_lm_trace,
            "marginal_linear": marginal_trace,
            "sequential_mixed_linear": linear_set_trace,
            "sequential_mixed_lambdamart": lm_set_trace,
        },
    }
    return metric_rows, prediction_rows, fold_audit


def _by_query_arm(rows: list[dict], metric: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for row in rows:
        value = row.get(metric)
        if value is not None:
            grouped[str(row["query_id"])][str(row["arm"])].append(float(value))
    return {
        qid: {
            arm: statistics.fmean(values)
            for arm, values in arms.items()
        }
        for qid, arms in grouped.items()
    }


def _aggregate_arm(
    rows: list[dict],
    arm: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    selected = [row for row in rows if row["arm"] == arm]
    metrics = (
        "mean_utility_at8",
        "ndcg_at8",
        "regret_to_full_oracle_at8",
        "selected_graph_count",
    )
    result = {
        "arm": arm,
        "oof_query_rows": len(selected),
        "unique_queries": len({row["query_id"] for row in selected}),
    }
    for metric in metrics:
        by_query = defaultdict(list)
        for row in selected:
            by_query[str(row["query_id"])].append(float(row[metric]))
        values = [statistics.fmean(items) for items in by_query.values()]
        lo, hi = bootstrap_ci(
            values, n_boot=bootstrap_samples, seed=bootstrap_seed + len(metric))
        result[metric] = {
            "mean": statistics.fmean(values),
            "query_bootstrap_95ci": [lo, hi],
        }
    graph_rows = [row for row in selected if row["include_graph"]]
    selected_graph = sum(int(row["selected_graph_count"]) for row in graph_rows)
    captured = sum(
        int(row["strict_graph_opportunity_captured"]) for row in graph_rows)
    opportunities = sum(
        int(row["strict_graph_opportunity_count"]) for row in graph_rows)
    result["graph_diagnostics"] = {
        "selected_graph_candidates": selected_graph,
        "strict_graph_opportunities": opportunities,
        "strict_graph_opportunities_captured": captured,
        "micro_graph_opportunity_recall": (
            captured / opportunities if opportunities else None),
        "micro_graph_entrant_precision": (
            captured / selected_graph if selected_graph else None),
        "query_opportunity_hit_rate": statistics.fmean(
            float(row["graph_opportunity_hit"])
            for row in graph_rows
            if row["graph_opportunity_hit"] is not None
        ) if any(
            row["graph_opportunity_hit"] is not None for row in graph_rows
        ) else None,
    }
    return result


def _paired_arm_contrast(
    rows: list[dict],
    left: str,
    right: str,
    metric: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    by_query = _by_query_arm(rows, metric)
    deltas = [
        arms[left] - arms[right]
        for arms in by_query.values()
        if left in arms and right in arms
    ]
    lo, hi = bootstrap_ci(
        deltas,
        n_boot=bootstrap_samples,
        seed=bootstrap_seed + len(left) + len(right) + len(metric),
    )
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "queries": len(deltas),
        "mean_delta": statistics.fmean(deltas),
        "query_bootstrap_95ci": [lo, hi],
        "improved_tied_degraded": [
            sum(value > 1e-9 for value in deltas),
            sum(abs(value) <= 1e-9 for value in deltas),
            sum(value < -1e-9 for value in deltas),
        ],
    }


def _resolve_config(config_key: str) -> tuple[Path, dict]:
    root = Path(__file__).resolve().parents[1]
    raw = dict(project_config.load()[config_key])
    for key in (
        "output_dir",
        "candidate_pool",
        "oracle_manifest",
        "utility_registry",
        "dense_run",
        "split_manifest",
    ):
        path = Path(raw[key])
        raw[key] = path if path.is_absolute() else root / path
        _reject_test(raw[key])
    raw["graph_runs"] = {
        name: (Path(path) if Path(path).is_absolute() else root / path)
        for name, path in raw["graph_runs"].items()
    }
    for path in raw["graph_runs"].values():
        _reject_test(path)
    return root, raw


def run(config_key: str = "strict_sbert_mixed_selector") -> dict:
    _, raw = _resolve_config(config_key)
    out_dir = raw["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_rows = _read_jsonl(raw["candidate_pool"])
    eligible_qids = {str(row["query_id"]) for row in pool_rows}
    dense = _run_features(raw["dense_run"], eligible_qids)
    graph_runs = {
        name: _run_features(path, eligible_qids)
        for name, path in raw["graph_runs"].items()
    }
    feature_rows = _build_mixed_feature_rows(pool_rows, dense, graph_runs)
    complete_rows, complete = complete_utility_v2_rows(
        _read_jsonl(raw["utility_registry"]))
    qrels = {
        key: float(row["utility"])
        for key, row in complete.items()
    }
    dataset = MixedDataset(feature_rows, qrels)
    split_manifest = json.loads(raw["split_manifest"].read_text(encoding="utf-8"))
    top_k = int(raw["top_k"])
    input_audit = _validate_inputs(
        dataset, split_manifest, top_k=top_k)

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    fold_audits: list[dict] = []
    for fold_row in sorted(
        split_manifest["rows"],
        key=lambda row: (int(row["repeat"]), int(row["fold"])),
    ):
        metrics, predictions, audit = _fold_predictions(
            dataset,
            fold_row,
            raw["training"],
            raw["selection"],
            dense_tail_start_rank=int(raw["dense_tail_start_rank"]),
            top_k=top_k,
        )
        metric_rows.extend(metrics)
        prediction_rows.extend(predictions)
        fold_audits.append(audit)
        print(json.dumps({
            "repeat": audit["repeat"],
            "fold": audit["fold"],
            "validation_queries": audit["validation_queries"],
            "marginal_accuracy": audit[
                "validation_marginal_pairwise_accuracy"],
            "linear_set": audit["linear_set_config"],
            "lambdamart_set": audit["lambdamart_set_config"],
        }, sort_keys=True), flush=True)

    arms = sorted({str(row["arm"]) for row in metric_rows})
    bootstrap_samples = int(raw["bootstrap_samples"])
    bootstrap_seed = int(raw["bootstrap_seed"])
    aggregate = {
        arm: _aggregate_arm(
            metric_rows,
            arm,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for arm in arms
    }
    matched_pairs = []
    model_stems = (
        "basic_linear",
        "mixed_linear",
        "mixed_lambdamart",
        "sequential_mixed_linear",
        "sequential_mixed_lambdamart",
    )
    for stem in model_stems:
        for metric in ("mean_utility_at8", "ndcg_at8"):
            matched_pairs.append(_paired_arm_contrast(
                metric_rows,
                f"{stem}_full",
                f"{stem}_depth",
                metric,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ))
    selector_pairs = []
    for left, right in (
        ("mixed_linear_full", "basic_linear_full"),
        ("mixed_lambdamart_full", "mixed_linear_full"),
        ("sequential_mixed_linear_full", "mixed_linear_full"),
        ("sequential_mixed_lambdamart_full", "mixed_lambdamart_full"),
        ("sequential_mixed_lambdamart_full", "dense_top8_raw"),
    ):
        for metric in ("mean_utility_at8", "ndcg_at8"):
            selector_pairs.append(_paired_arm_contrast(
                metric_rows,
                left,
                right,
                metric,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ))

    oracle_manifest = json.loads(
        raw["oracle_manifest"].read_text(encoding="utf-8"))
    oracle_graph_gain = float(
        oracle_manifest["metrics"]["graph_gain_beyond_depth"]["mean"])
    graph_conversions = []
    for contrast in matched_pairs:
        if contrast["metric"] != "mean_utility_at8":
            continue
        graph_conversions.append({
            "arm": contrast["left"].removesuffix("_full"),
            "deployable_graph_gain": contrast["mean_delta"],
            "oracle_graph_gain_beyond_depth": oracle_graph_gain,
            "oracle_conversion_ratio": (
                contrast["mean_delta"] / oracle_graph_gain
                if oracle_graph_gain else None
            ),
            "query_bootstrap_95ci": contrast["query_bootstrap_95ci"],
        })
    marginal_accuracy = statistics.fmean(
        float(row["validation_marginal_pairwise_accuracy"])
        for row in fold_audits
    )
    marginal_distribution = _marginal_label_distribution(
        dataset,
        sorted(dataset.by_query),
        dense_tail_start_rank=int(raw["dense_tail_start_rank"]),
        margin=float(raw["training"]["pair_margin"]),
    )
    marginal_distribution["oof_query_macro_pairwise_accuracy"] = (
        marginal_accuracy)
    marginal_distribution["accuracy_lift_over_always_dense_query_macro"] = (
        marginal_accuracy
        - marginal_distribution["query_macro_always_dense_accuracy"]
    )
    report = {
        "schema": "strict-sbert-mixed-selector-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "development-only; fully judged LLM-silver utility-v2",
        "version": raw["version"],
        "verdict": (
            "STRICT_SBERT_GRAPH_HEADROOM_NOT_CONVERTED_BY_TESTED_SELECTORS"),
        "input_audit": input_audit,
        "feature_contract": {
            "basic_features": list(BASIC_FEATURES),
            "mixed_route_features": list(ROUTE_FEATURES),
            "normalization": (
                "StandardScaler fitted inside each inner/outer training fold; "
                "run scores are query-local min-max normalized"),
            "candidate_pool_identical_for_all_trained_models": True,
            "graph_vs_depth_changes_candidate_availability_only": True,
            "utility_used_for_candidate_or_feature_construction": False,
        },
        "training_contract": {
            "outer_split": "existing development200 grouped 5x5 folds",
            "hyperparameter_selection": "outer-train only",
            "pointwise_families": ["pairwise Linear", "XGBoost LambdaMART"],
            "marginal_target": (
                "graph candidates vs Dense Top-8 ranks 7-8 and Depth4; "
                "pairs with absolute utility difference below margin omitted"),
            "set_selector": (
                "greedy protected-Dense selection with lexical redundancy "
                "penalty and graph-only marginal score; parameter grid "
                "selected on outer-train only"),
            "full_SetR_or_JPR_reproduction": False,
            "test_used": False,
        },
        "marginal_pairwise_accuracy_oof_fold_mean": marginal_accuracy,
        "marginal_directional_diagnostic": marginal_distribution,
        "arms": aggregate,
        "matched_graph_availability_contrasts": matched_pairs,
        "selector_contrasts": selector_pairs,
        "oracle_conversion": graph_conversions,
        "inputs": {
            key: {"path": str(raw[key].resolve()), "sha256": _sha256(raw[key])}
            for key in (
                "candidate_pool",
                "oracle_manifest",
                "utility_registry",
                "dense_run",
                "split_manifest",
            )
        },
        "graph_runs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in raw["graph_runs"].items()
        },
        "utility_registry_complete_rows": len(complete_rows),
        "claim_boundary": (
            "OOF development LLM-silver evidence selection only. A positive "
            "matched graph contrast shows recoverable candidate utility in "
            "this fixed pool; it is not frozen-test or answer-level evidence."
        ),
    }
    _write_jsonl(out_dir / "mixed_candidate_feature_registry.jsonl", feature_rows)
    _write_jsonl(out_dir / "oof_query_metrics.jsonl", metric_rows)
    _write_jsonl(out_dir / "oof_rankings.jsonl", prediction_rows)
    _write_json(out_dir / "nested_fold_audit.json", {"folds": fold_audits})
    _write_json(out_dir / "strict_sbert_mixed_selector_report.json", report)
    _write_json(out_dir / "manifest.json", {
        "schema": report["schema"],
        "created_at": report["created_at"],
        "version": raw["version"],
        "status": "COMPLETE",
        "queries": input_audit["queries"],
        "candidate_pairs": input_audit["candidate_pairs"],
        "oof_query_rows": len(metric_rows),
        "prediction_rows": len(prediction_rows),
        "folds": len(fold_audits),
        "external_calls": 0,
        "test_read": False,
        "report": "strict_sbert_mixed_selector_report.json",
    })
    return report


def refresh_existing_diagnostics(
    config_key: str = "strict_sbert_mixed_selector",
) -> dict:
    """Refresh label-distribution diagnostics without retraining models."""
    _, raw = _resolve_config(config_key)
    report_path = (
        raw["output_dir"] / "strict_sbert_mixed_selector_report.json")
    feature_path = (
        raw["output_dir"] / "mixed_candidate_feature_registry.jsonl")
    if not report_path.exists() or not feature_path.exists():
        raise FileNotFoundError(
            "completed report and feature registry are required")
    _, complete = complete_utility_v2_rows(
        _read_jsonl(raw["utility_registry"]))
    qrels = {
        key: float(row["utility"])
        for key, row in complete.items()
    }
    dataset = MixedDataset(_read_jsonl(feature_path), qrels)
    diagnostic = _marginal_label_distribution(
        dataset,
        sorted(dataset.by_query),
        dense_tail_start_rank=int(raw["dense_tail_start_rank"]),
        margin=float(raw["training"]["pair_margin"]),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["verdict"] = (
        "STRICT_SBERT_GRAPH_HEADROOM_NOT_CONVERTED_BY_TESTED_SELECTORS")
    marginal_accuracy = float(
        report["marginal_pairwise_accuracy_oof_fold_mean"])
    diagnostic["oof_query_macro_pairwise_accuracy"] = marginal_accuracy
    diagnostic["accuracy_lift_over_always_dense_query_macro"] = (
        marginal_accuracy
        - diagnostic["query_macro_always_dense_accuracy"]
    )
    report["marginal_directional_diagnostic"] = diagnostic
    _write_json(report_path, report)
    return diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-key", default="strict_sbert_mixed_selector")
    parser.add_argument(
        "--refresh-diagnostics-only", action="store_true",
        help="Update class-imbalance diagnostics in an existing completed report.",
    )
    args = parser.parse_args()
    if args.refresh_diagnostics_only:
        print(json.dumps(
            refresh_existing_diagnostics(args.config_key),
            ensure_ascii=False,
            indent=2,
        ))
        return
    report = run(args.config_key)
    print(json.dumps({
        "status": "COMPLETE",
        "queries": report["input_audit"]["queries"],
        "marginal_pairwise_accuracy": report[
            "marginal_pairwise_accuracy_oof_fold_mean"],
        "oracle_conversion": report["oracle_conversion"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
