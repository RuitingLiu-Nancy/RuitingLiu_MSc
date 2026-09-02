#!/usr/bin/env python3
"""Coverage-complete Dense M<=50 frontier with the frozen Report-91 selector.

The command is development100-only and local-only.  It consumes the frozen
MiniLM/E5 D100 rankings only up to M=50, the non-destructive 19,813-row
utility-v2 registry, and the exact Report-89 5x5 query-grouped splits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from evaluation.judgment_completeness import (
        complete_utility_v2_rows,
    )
    from evaluation.statistics import bootstrap_ci
    from utility_scoring.learned_diffusion import reranker_validation as canonical
    from candidate_pool.run_dense_semantic_drift_rescue_audit import (
        SOURCE_BLIND_FEATURES,
        STATIC_PREDICTOR_FEATURES,
        _action_dimension_decomposition,
        _build_idf,
        _cosine,
        _evaluate_fold_variant,
        _fit_auxiliary_utility_model,
        _fold_local_predicted_utility,
        _lexical_diagnostics,
        _load_embeddings,
        _tokenize,
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
    from evaluation.judgment_completeness import (
        complete_utility_v2_rows,
    )
    from evaluation.statistics import bootstrap_ci
    from utility_scoring.learned_diffusion import reranker_validation as canonical
    from candidate_pool.run_dense_semantic_drift_rescue_audit import (
        SOURCE_BLIND_FEATURES,
        STATIC_PREDICTOR_FEATURES,
        _action_dimension_decomposition,
        _build_idf,
        _cosine,
        _evaluate_fold_variant,
        _fit_auxiliary_utility_model,
        _fold_local_predicted_utility,
        _lexical_diagnostics,
        _load_embeddings,
        _tokenize,
    )
    from utility_scoring.annotation.run_top3_residual_judging import (
        read_jsonl,
        sha256,
        utc_now,
        write_json,
        write_jsonl,
    )


DEPTHS = (8, 12, 20, 50)
BACKENDS = ("minilm", "e5")
TOP_K = 8
USEFUL_THRESHOLD = 4.0
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20260730
AUX_CONFIG = {"huber_epsilon": 1.35, "l2_alpha": 0.1, "max_iter": 1000}
DIRECT_CONFIG = {"huber_epsilon": 1.35, "l2_alpha": 0.1, "max_iter": 1000}
KAPPA_OPTIONS = (1.0, 2.0, 4.0)
THRESHOLD_QUANTILES = (0.25, 0.5, 0.75, 0.9)


def load_rankings(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        grouped[str(row["query_id"])].append(row)
    for query_id in grouped:
        grouped[query_id].sort(key=lambda row: int(row["rank"]))
        if [int(row["rank"]) for row in grouped[query_id]] != list(
            range(1, len(grouped[query_id]) + 1)
        ):
            raise ValueError(f"{path}: non-contiguous ranks for {query_id}")
    return dict(grouped)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def static_features_for_arm(
    *,
    query_ids: list[str],
    baseline_ids: dict[str, list[str]],
    arm_ids: dict[str, list[str]],
    rank_maps: dict[str, dict[str, int]],
    candidate_vectors: dict[str, np.ndarray],
    query_vectors: dict[str, np.ndarray],
    corpus_text: dict[str, str],
    query_text: dict[str, str],
    idf: dict[str, float],
) -> dict[tuple[str, str], dict[str, float]]:
    output: dict[tuple[str, str], dict[str, float]] = {}
    for query_id in query_ids:
        d8_vectors = [candidate_vectors[cid] for cid in baseline_ids[query_id]]
        for candidate_id in arm_ids[query_id]:
            vector = candidate_vectors[candidate_id]
            similarities = [_cosine(vector, other) for other in d8_vectors]
            lexical = _lexical_diagnostics(
                query_text[query_id],
                corpus_text[candidate_id],
                idf,
            )
            rank = rank_maps[query_id].get(candidate_id)
            output[(query_id, candidate_id)] = {
                "query_candidate_sbert_similarity": _cosine(
                    query_vectors[query_id],
                    vector,
                ),
                "candidate_dense_percentile": (
                    (101 - rank) / 100.0 if rank is not None else 0.0
                ),
                "candidate_dense_rank_missing": float(rank is None),
                "comment_length_log": math.log1p(
                    len(_tokenize(corpus_text[candidate_id]))
                ),
                "idf_weighted_lexical_overlap": lexical[
                    "idf_weighted_lexical_overlap"
                ],
                "candidate_to_d8_max_similarity": max(similarities),
                "candidate_to_d8_mean_similarity": statistics.fmean(similarities),
                "candidate_novelty_relative_to_d8": 1.0 - max(similarities),
            }
    if any(tuple(row) != STATIC_PREDICTOR_FEATURES for row in output.values()):
        raise AssertionError("static feature order drifted")
    return output


def build_action_features(
    *,
    query_id: str,
    baseline: list[str],
    candidate_id: str,
    replaced_id: str,
    predicted: dict[tuple[str, str], float],
    static: dict[tuple[str, str], dict[str, float]],
    candidate_vectors: dict[str, np.ndarray],
) -> dict[str, float]:
    """Canonical source-blind one-swap features, with no outcome access."""
    baseline_vectors = [candidate_vectors[cid] for cid in baseline]
    pairwise = [
        _cosine(baseline_vectors[left], baseline_vectors[right])
        for left in range(len(baseline_vectors))
        for right in range(left + 1, len(baseline_vectors))
    ]
    candidate_static = static[(query_id, candidate_id)]
    replaced_static = static[(query_id, replaced_id)]
    features = {
        "query_candidate_sbert_similarity": candidate_static[
            "query_candidate_sbert_similarity"
        ],
        "candidate_dense_percentile": candidate_static[
            "candidate_dense_percentile"
        ],
        "candidate_dense_rank_missing": candidate_static[
            "candidate_dense_rank_missing"
        ],
        "comment_length_log": candidate_static["comment_length_log"],
        "idf_weighted_lexical_overlap": candidate_static[
            "idf_weighted_lexical_overlap"
        ],
        "candidate_to_d8_max_similarity": candidate_static[
            "candidate_to_d8_max_similarity"
        ],
        "candidate_to_d8_mean_similarity": candidate_static[
            "candidate_to_d8_mean_similarity"
        ],
        "candidate_novelty_relative_to_d8": candidate_static[
            "candidate_novelty_relative_to_d8"
        ],
        "candidate_to_replaced_item_similarity": _cosine(
            candidate_vectors[candidate_id], candidate_vectors[replaced_id]
        ),
        "oof_predicted_utility_difference": (
            predicted[(query_id, candidate_id)]
            - predicted[(query_id, replaced_id)]
        ),
        "query_similarity_difference": (
            candidate_static["query_candidate_sbert_similarity"]
            - replaced_static["query_candidate_sbert_similarity"]
        ),
        "comment_length_difference": (
            candidate_static["comment_length_log"]
            - replaced_static["comment_length_log"]
        ),
        "idf_overlap_difference": (
            candidate_static["idf_weighted_lexical_overlap"]
            - replaced_static["idf_weighted_lexical_overlap"]
        ),
        "d8_similarity_dispersion": float(np.std(pairwise)),
        "d8_redundancy": statistics.fmean(pairwise),
    }
    if tuple(features) != SOURCE_BLIND_FEATURES:
        raise AssertionError("source-blind feature order drifted")
    return features


def build_actions(
    *,
    repeat: int,
    fold: int,
    train_ids: list[str],
    valid_ids: list[str],
    pool: str,
    baseline_ids: dict[str, list[str]],
    candidate_ids: dict[str, list[str]],
    predicted: dict[tuple[str, str], float],
    static: dict[tuple[str, str], dict[str, float]],
    candidate_vectors: dict[str, np.ndarray],
    registry: dict[tuple[str, str], dict],
) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    contexts = {}
    for query_id in train_ids + valid_ids:
        baseline = baseline_ids[query_id]
        baseline_utility = statistics.fmean(
            float(registry[(query_id, cid)]["utility"]) for cid in baseline
        )
        contexts[query_id] = {"current_full_utility": baseline_utility}
        rows.append({
            "repeat": repeat,
            "fold": fold,
            "query_id": query_id,
            "pool": pool,
            "action_id": "NOOP",
            "action_type": "NOOP",
            "candidate_id": None,
            "replaced_candidate_id": None,
            "replacement_rank": None,
            "eligible_all8": True,
            "model_features": {name: 0.0 for name in SOURCE_BLIND_FEATURES},
            "raw_reward_for_training_and_evaluation_only": 0.0,
            "baseline_utility_for_reward_only": baseline_utility,
            "action_utility_for_reward_only": baseline_utility,
            "fallback_used": False,
            "callback_used": False,
            "padding_used": False,
            "route_membership": "NOOP",
            "source_attribution_traceable": True,
            "inference_used_gold_utility": False,
        })
        for candidate_id in candidate_ids[query_id]:
            for replacement_rank, replaced_id in enumerate(baseline, start=1):
                features = build_action_features(
                    query_id=query_id,
                    baseline=baseline,
                    candidate_id=candidate_id,
                    replaced_id=replaced_id,
                    predicted=predicted,
                    static=static,
                    candidate_vectors=candidate_vectors,
                )
                decomposition = _action_dimension_decomposition(
                    registry[(query_id, candidate_id)],
                    registry[(query_id, replaced_id)],
                )
                reward = float(decomposition["raw_utility_at8_delta"])
                rows.append({
                    "repeat": repeat,
                    "fold": fold,
                    "query_id": query_id,
                    "pool": pool,
                    "action_id": f"replace:r{replacement_rank}:{candidate_id}",
                    "action_type": "REPLACE",
                    "candidate_id": candidate_id,
                    "replaced_candidate_id": replaced_id,
                    "replacement_rank": replacement_rank,
                    "eligible_all8": True,
                    "model_features": features,
                    "raw_reward_for_training_and_evaluation_only": reward,
                    "baseline_utility_for_reward_only": baseline_utility,
                    "action_utility_for_reward_only": baseline_utility + reward,
                    "fallback_used": False,
                    "callback_used": False,
                    "padding_used": False,
                    "route_membership": pool,
                    "source_attribution_traceable": True,
                    "inference_used_gold_utility": False,
                    **decomposition,
                })
    return rows, contexts


def run_selector_arm(
    *,
    backend: str,
    depth: int,
    query_ids: list[str],
    split_rows: list[dict],
    baseline_ids: dict[str, list[str]],
    candidate_ids: dict[str, list[str]],
    static: dict[tuple[str, str], dict[str, float]],
    candidate_vectors: dict[str, np.ndarray],
    registry: dict[tuple[str, str], dict],
    pool_name: str | None = None,
) -> tuple[list[dict], list[dict]]:
    if not any(candidate_ids.values()):
        rows = []
        for split in split_rows:
            for query_id in map(str, split["validation_query_ids"]):
                baseline = statistics.fmean(
                    float(registry[(query_id, cid)]["utility"])
                    for cid in baseline_ids[query_id]
                )
                rows.append({
                    "repeat": int(split["repeat"]),
                    "fold": int(split["fold"]),
                    "query_id": query_id,
                    "backend": backend,
                    "depth": depth,
                    "threshold_mode": "nested",
                    "acted": False,
                    "successful_action": False,
                    "harmful_action": False,
                    "raw_reward": 0.0,
                    "baseline_utility_at8": baseline,
                    "policy_utility_at8": baseline,
                    "action_space_oracle_headroom": 0.0,
                    "selected_candidate_id": None,
                    "replaced_candidate_id": None,
                    "predicted_margin": None,
                    "inference_used_gold_utility": False,
                })
        return rows, []

    pool = pool_name or f"{backend}_D{depth}_tail"
    combined = {
        query_id: [*baseline_ids[query_id], *candidate_ids[query_id]]
        for query_id in query_ids
    }
    output, audit = [], []
    validation_count = Counter()
    for split in split_rows:
        repeat = int(split["repeat"])
        fold = int(split["fold"])
        seed = int(split["seed"])
        train_ids = list(map(str, split["train_query_ids"]))
        valid_ids = list(map(str, split["validation_query_ids"]))
        validation_count.update(valid_ids)
        inner = canonical.inner_folds(train_ids, 3, seed + 7000)
        predicted, prediction_audit = _fold_local_predicted_utility(
            full_train_qids=train_ids,
            full_valid_qids=valid_ids,
            inner_splits=inner,
            eligible_qids=set(query_ids),
            candidate_ids=combined,
            static_features=static,
            registry=registry,
            config=AUX_CONFIG,
        )
        actions, contexts = build_actions(
            repeat=repeat,
            fold=fold,
            train_ids=train_ids,
            valid_ids=valid_ids,
            pool=pool,
            baseline_ids=baseline_ids,
            candidate_ids=candidate_ids,
            predicted=predicted,
            static=static,
            candidate_vectors=candidate_vectors,
            registry=registry,
        )
        predictions, tuning = _evaluate_fold_variant(
            actions,
            pool=pool,
            action_space="all8",
            family="direct_delta",
            train_qids=train_ids,
            valid_qids=valid_ids,
            inner_splits=inner,
            model_config=DIRECT_CONFIG,
            kappa_options=list(KAPPA_OPTIONS),
            threshold_quantiles=list(THRESHOLD_QUANTILES),
            seed=seed,
            baseline_contexts=contexts,
            feature_names=SOURCE_BLIND_FEATURES,
        )
        for row in predictions:
            if row["threshold_mode"] == "nested":
                row["backend"] = backend
                row["depth"] = depth
                output.append(row)
        audit.append({
            "backend": backend,
            "depth": depth,
            "repeat": repeat,
            "fold": fold,
            "train_queries": len(train_ids),
            "validation_queries": len(valid_ids),
            "train_validation_overlap": len(set(train_ids) & set(valid_ids)),
            "prediction_audit": prediction_audit,
            "tuning": tuning,
        })
    if set(validation_count) != set(query_ids) or any(
        count != 5 for count in validation_count.values()
    ):
        raise ValueError("outer validation membership changed")
    return output, audit


def aggregate_metrics(
    *,
    backend: str,
    depth: int,
    rankings: dict[str, list[dict]],
    oof_rows: list[dict],
    registry: dict[tuple[str, str], dict],
) -> tuple[dict, list[dict], list[dict]]:
    query_ids = sorted(rankings)
    candidate_rows = [
        registry[(query_id, str(row["comment_id"]))]
        for query_id in query_ids
        for row in rankings[query_id][:depth]
    ]
    candidate_utilities = [float(row["utility"]) for row in candidate_rows]
    baseline, oracle, base_rate_rows = {}, {}, []
    for query_id in query_ids:
        pool_ids = [
            str(row["comment_id"]) for row in rankings[query_id][:depth]
        ]
        d8 = pool_ids[:TOP_K]
        baseline[query_id] = statistics.fmean(
            float(registry[(query_id, cid)]["utility"]) for cid in d8
        )
        oracle_ids = sorted(
            pool_ids,
            key=lambda cid: (
                float(registry[(query_id, cid)]["utility"]),
                -pool_ids.index(cid),
                cid,
            ),
            reverse=True,
        )[:TOP_K]
        oracle[query_id] = statistics.fmean(
            float(registry[(query_id, cid)]["utility"]) for cid in oracle_ids
        )
        for candidate_id in pool_ids[TOP_K:]:
            for replaced_id in d8:
                delta = (
                    float(registry[(query_id, candidate_id)]["utility"])
                    - float(registry[(query_id, replaced_id)]["utility"])
                ) / TOP_K
                base_rate_rows.append({
                    "backend": backend,
                    "depth": depth,
                    "query_id": query_id,
                    "candidate_id": candidate_id,
                    "replaced_candidate_id": replaced_id,
                    "raw_utility_at8_delta": delta,
                    "positive": delta > 0,
                    "negative": delta < 0,
                    "tie": delta == 0,
                })
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in oof_rows:
        by_query[str(row["query_id"])].append(row)
    realised = {
        query_id: statistics.fmean(
            float(row["policy_utility_at8"]) for row in by_query[query_id]
        )
        for query_id in query_ids
    }
    deltas = [realised[qid] - baseline[qid] for qid in query_ids]
    headroom = [oracle[qid] - baseline[qid] for qid in query_ids]
    actions = [
        statistics.fmean(float(row["acted"]) for row in by_query[qid])
        for qid in query_ids
    ]
    successes = [
        statistics.fmean(float(row["successful_action"]) for row in by_query[qid])
        for qid in query_ids
    ]
    harms = [
        statistics.fmean(float(row["harmful_action"]) for row in by_query[qid])
        for qid in query_ids
    ]
    action_mass, success_mass, harm_mass = sum(actions), sum(successes), sum(harms)
    opportunity_queries = sum(value > 0 for value in headroom)
    lo, hi = bootstrap_ci(deltas, n_boot=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED + depth)
    metrics = {
        "backend": backend,
        "depth": depth,
        "queries": len(query_ids),
        "candidate_pairs": len(candidate_rows),
        "candidate_mean_utility": statistics.fmean(candidate_utilities),
        "candidate_median_utility": statistics.median(candidate_utilities),
        "candidate_useful_rate": statistics.fmean(
            value >= USEFUL_THRESHOLD for value in candidate_utilities
        ),
        "candidate_safety_gated_rate": statistics.fmean(
            float(row["label_safety"]) <= 2 for row in candidate_rows
        ),
        "baseline_utility_at8": statistics.fmean(baseline.values()),
        "oracle_utility_at8": statistics.fmean(oracle.values()),
        "marginal_oracle_headroom": statistics.fmean(headroom),
        "realised_oof_utility_at8": statistics.fmean(realised.values()),
        "realised_delta_vs_d8": statistics.fmean(deltas),
        "realised_delta_query_bootstrap_95ci_lo": lo,
        "realised_delta_query_bootstrap_95ci_hi": hi,
        "action_rate": statistics.fmean(actions),
        "entrant_precision": success_mass / action_mass if action_mass else None,
        "harmful_action_rate": statistics.fmean(harms),
        "opportunity_queries": opportunity_queries,
        "opportunity_recall": (
            success_mass / opportunity_queries if opportunity_queries else None
        ),
        "conversion_ratio": (
            statistics.fmean(deltas) / statistics.fmean(headroom)
            if statistics.fmean(headroom) else None
        ),
        "regret_to_oracle": statistics.fmean(
            oracle[qid] - realised[qid] for qid in query_ids
        ),
        "positive_action_base_rate": statistics.fmean(
            row["positive"] for row in base_rate_rows
        ) if base_rate_rows else 0.0,
        "negative_action_base_rate": statistics.fmean(
            row["negative"] for row in base_rate_rows
        ) if base_rate_rows else 0.0,
        "successful_action_mass": success_mass,
        "harmful_action_mass": harm_mass,
        "selector_family": "Report91 Direct Huber source-blind one-swap with NOOP",
    }
    per_query = [{
        "backend": backend,
        "depth": depth,
        "query_id": qid,
        "baseline_utility_at8": baseline[qid],
        "oracle_utility_at8": oracle[qid],
        "oracle_headroom": oracle[qid] - baseline[qid],
        "realised_oof_utility_at8": realised[qid],
        "realised_delta": realised[qid] - baseline[qid],
        "action_rate": statistics.fmean(
            float(row["acted"]) for row in by_query[qid]
        ),
        "successful_action_rate": statistics.fmean(
            float(row["successful_action"]) for row in by_query[qid]
        ),
        "harmful_action_rate": statistics.fmean(
            float(row["harmful_action"]) for row in by_query[qid]
        ),
    } for qid in query_ids]
    return metrics, per_query, base_rate_rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "out/depth_graph_utility_community_frontier_dev100_m50_dense_v1",
    )
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)

    preflight = root / "out/depth_graph_utility_community_frontier_dev100_m50_v2"
    registry_path = (
        root
        / "out/depth_graph_utility_community_frontier_dev100_m50_judging_v1"
        / "complete/utility_registry_coverage_complete.jsonl"
    )
    registry_rows = read_jsonl(registry_path)
    complete, registry = complete_utility_v2_rows(registry_rows)
    if len(complete) != 19813:
        raise ValueError(f"expected 19,813 complete rows, got {len(complete)}")

    ranking_paths = {
        "minilm": preflight / "m50_residual_preflight/dense_m50_memberships.jsonl",
        "e5": preflight / "m50_residual_preflight/dense_m50_memberships.jsonl",
    }
    membership_rows = read_jsonl(ranking_paths["minilm"])
    rankings: dict[str, dict[str, list[dict]]] = {
        backend: defaultdict(list) for backend in BACKENDS
    }
    for row in membership_rows:
        rankings[str(row["backend"])][str(row["query_id"])].append(row)
    for backend in BACKENDS:
        rankings[backend] = dict(rankings[backend])
        for query_id in rankings[backend]:
            rankings[backend][query_id].sort(key=lambda row: int(row["rank"]))
            if len(rankings[backend][query_id]) != 50:
                raise ValueError(f"{backend}/{query_id}: expected D50")

    split_path = (
        root / "out/strict_native_graph_conservative_policy_dev100_v1/split_manifest.json"
    )
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    split_rows = split_manifest["rows"]
    query_ids = sorted(rankings["minilm"])
    if len(query_ids) != 100 or set(query_ids) != set(rankings["e5"]):
        raise ValueError("development100 query identity mismatch")

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
    idf, idf_manifest = _build_idf(corpus_text.values())

    metrics_rows, per_query_rows, base_rate_rows = [], [], []
    all_oof, fold_audit = [], []
    for backend in BACKENDS:
        rank_maps = {
            qid: {
                str(row["comment_id"]): int(row["rank"])
                for row in rankings[backend][qid]
            }
            for qid in query_ids
        }
        baseline_ids = {
            qid: [
                str(row["comment_id"]) for row in rankings[backend][qid][:TOP_K]
            ]
            for qid in query_ids
        }
        for depth in DEPTHS:
            candidate_ids = {
                qid: [
                    str(row["comment_id"])
                    for row in rankings[backend][qid][TOP_K:depth]
                ]
                for qid in query_ids
            }
            arm_ids = {
                qid: [*baseline_ids[qid], *candidate_ids[qid]]
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
                split_rows=split_rows,
                baseline_ids=baseline_ids,
                candidate_ids=candidate_ids,
                static=static,
                candidate_vectors=candidate_vectors,
                registry=registry,
            )
            metrics, per_query, base_rates = aggregate_metrics(
                backend=backend,
                depth=depth,
                rankings=rankings[backend],
                oof_rows=oof,
                registry=registry,
            )
            metrics_rows.append(metrics)
            per_query_rows.extend(per_query)
            base_rate_rows.extend(base_rates)
            all_oof.extend(oof)
            fold_audit.extend(audit)
            print(json.dumps({
                "backend": backend,
                "depth": depth,
                "realised_delta": metrics["realised_delta_vs_d8"],
                "oracle_headroom": metrics["marginal_oracle_headroom"],
            }), flush=True)

    write_csv(out_dir / "m50_depth_metrics.csv", metrics_rows)
    write_csv(out_dir / "m50_selection_burden.csv", metrics_rows)
    write_csv(out_dir / "m50_dense_per_query.csv", per_query_rows)
    write_jsonl(out_dir / "m50_dense_oof_actions.jsonl", all_oof)
    write_jsonl(out_dir / "m50_dense_action_base_rates.jsonl", base_rate_rows)
    write_json(out_dir / "nested_fold_audit.json", fold_audit)
    write_json(out_dir / "idf_manifest.json", idf_manifest)
    input_paths = [
        registry_path,
        preflight / "manifest.json",
        preflight / "m50_residual_preflight/dense_m50_memberships.jsonl",
        split_path,
        root / "out/sbert_graph_comparison_dev100_v1/sbert_corpus_embeddings.npy",
        root / "out/sbert_graph_comparison_dev100_v1/sbert_query_embeddings.npy",
    ]
    output_paths = sorted(
        path for path in out_dir.iterdir() if path.is_file()
    )
    manifest = {
        "schema": "dev100-m50-dense-frontier-analysis-v1",
        "created_utc": utc_now(),
        "status": "DENSE_M50_FRONTIER_COMPLETE",
        "development_queries": 100,
        "depths": list(DEPTHS),
        "backends": list(BACKENDS),
        "selector": {
            "family": "HuberRegressor Direct Delta Regression",
            "source_blind_features": list(SOURCE_BLIND_FEATURES),
            "NOOP": True,
            "maximum_swaps_per_query": 1,
            "outer_splits": "exact Report89 5x5 query-grouped",
            "inner_folds": 3,
            "kappa_options": list(KAPPA_OPTIONS),
            "threshold_quantiles": list(THRESHOLD_QUANTILES),
            "gold_utility_in_inference_features": False,
        },
        "utility_registry_rows": len(complete),
        "frozen_test_read": False,
        "m100_analysed": False,
        "remaining_development298_accessed": False,
        "input_hashes": {str(path.relative_to(root)): sha256(path) for path in input_paths},
        "output_hashes": {path.name: sha256(path) for path in output_paths},
    }
    write_json(out_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
