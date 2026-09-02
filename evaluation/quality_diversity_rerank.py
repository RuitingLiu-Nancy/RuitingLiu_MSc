"""Validation-only quality/diversity reranking over the labelled deep pool.

This module reuses evidence-level LLM utility labels as a *quality* signal and
keeps diversity as a list-level property.  It deliberately does not turn ILD
into a pointwise label: a candidate is only redundant relative to items already
selected for the same query.

The oracle utility arms in this script are diagnostic upper bounds.  In the
final system ``utility`` must be replaced by an out-of-fold prediction from a
query-grouped reranker; the same gate + MMR selector can then be reused.

Example:
  python -m evaluation.quality_diversity_rerank \
    --study-db ../study_platform/backend/data/study.db \
    --heldout out/expanded_graph_rebuild/heldout_validation.csv \
    --out-dir out/quality_diversity_llm_pilot
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation.sampling.sample_human_annotation_candidates import _load_cards
from evaluation.ir_metrics import graded_ndcg_at
from evaluation.statistics import bootstrap_ci
import configuration as config


TEXT_FEATURES = (
    "semantic_rr", "from_semantic", "fusion_rr", "from_fusion",
    "best_rr", "n_sources", "lexical_similarity", "snippet_len_words",
    "domain_count", "ef_count", "support_count",
)
GRAPH_FEATURES = TEXT_FEATURES + (
    "graph_rr", "from_graph", "graph_only", "entity_count",
)
OBSERVED_GRAPH_FEATURES = (
    "graph_score", "observed_min_hop",
    "observed_seed_entity_connection_count", "observed_reachable_seed_count",
    "observed_max_intermediate_degree", "observed_mean_intermediate_degree",
    "observed_hub_intermediate_count", "direct_seed_match",
    "relation_mediated", "observed_mentioned_entity_count",
    "unique_graph_retrieval",
)
RICH_GRAPH_FEATURES = GRAPH_FEATURES + OBSERVED_GRAPH_FEATURES


def _reasonable(row: dict, relevance: float, usefulness: float, safety: float) -> bool:
    return (
        float(row["label_relevance"]) >= relevance
        and float(row["label_usefulness"]) >= usefulness
        and float(row["label_safety"]) >= safety
    )


def _ild(selected: list[int], vectors) -> float:
    if len(selected) < 2:
        return 0.0
    block = vectors[selected]
    sim = (block @ block.T).toarray()
    upper = sim[np.triu_indices(len(selected), 1)]
    return float(1.0 - upper.mean()) if len(upper) else 0.0


def _mmr_select(
    rows: list[dict],
    vectors,
    candidate_ids: list[int],
    k: int,
    diversity_lambda: float,
    quality_values: list[float] | None = None,
) -> list[int]:
    """Greedy MMR where lambda is the diversity penalty weight.

    quality is fixed to [0, 1] from the 1--7 utility scale.  Redundancy is the
    maximum text cosine to an already selected item.  At lambda=0 this exactly
    reduces to utility ordering.
    """
    remaining = set(candidate_ids)
    selected: list[int] = []
    values = quality_values or [float(row["utility"]) for row in rows]
    quality = {i: max(0.0, min(1.0, (float(values[i]) - 1.0) / 6.0))
               for i in candidate_ids}
    while remaining and len(selected) < k:
        best_i = None
        best_key = None
        for i in remaining:
            redundancy = 0.0
            if selected:
                redundancy = float((vectors[i] @ vectors[selected].T).toarray().max())
            score = (1.0 - diversity_lambda) * quality[i] - diversity_lambda * redundancy
            key = (score, quality[i], -i)
            if best_key is None or key > best_key:
                best_i, best_key = i, key
        selected.append(best_i)  # type: ignore[arg-type]
        remaining.remove(best_i)  # type: ignore[arg-type]
    return selected


def _rr(rank: int) -> float:
    return 1.0 / (60.0 + rank) if rank else 0.0


def _load_observed_graph_features(path: Path | None) -> dict[tuple[str, str], dict]:
    if path is None:
        return {}
    if "test" in path.name.lower():
        raise ValueError("frozen test feature files are not accepted")
    out = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (str(row["query_id"]), str(row["comment_id"]))
            if key in out:
                raise ValueError(f"duplicate graph-feature key: {key}")
            out[key] = {name: float(row[name]) for name in OBSERVED_GRAPH_FEATURES}
    return out


def _feature_rows(cards: list[dict], snippet_vectors, tfidf,
                  observed_graph: dict[tuple[str, str], dict] | None = None) -> list[dict]:
    observed_graph = observed_graph or {}
    query_vectors = tfidf.transform([row["query_text"] for row in cards])
    lexical = np.asarray(query_vectors.multiply(snippet_vectors).sum(axis=1)).ravel()
    out = []
    for i, row in enumerate(cards):
        ranks = [int(row[key]) for key in (
            "semantic_rank", "fusion_dense_bm25_rank", "graph_rank") if int(row[key])]
        row_features = {
            "semantic_rr": _rr(int(row["semantic_rank"])),
            "from_semantic": int(bool(row["semantic_rank"])),
            "fusion_rr": _rr(int(row["fusion_dense_bm25_rank"])),
            "from_fusion": int(bool(row["fusion_dense_bm25_rank"])),
            "graph_rr": _rr(int(row["graph_rank"])),
            "from_graph": int(bool(row["graph_rank"])),
            "best_rr": _rr(min(ranks) if ranks else 0),
            "n_sources": int(row["n_sources"]),
            "graph_only": int(row["graph_only"]),
            "lexical_similarity": float(lexical[i]),
            "snippet_len_words": len(row["snippet"].split()),
            "domain_count": int(row["domain_count"]),
            "ef_count": int(row["ef_count"]),
            "support_count": int(row["support_count"]),
            "entity_count": int(row["entity_count"]),
        }
        key = (str(row["query_id"]), str(row["comment_id"]))
        if observed_graph:
            if key not in observed_graph:
                raise ValueError(f"missing observed graph features for card key: {key}")
            row_features.update(observed_graph[key])
        out.append(row_features)
    return out


def _oof_rf(
    features: list[dict],
    feature_names: tuple[str, ...],
    target: list[float],
    groups: list[str],
    folds: int,
    trees: int,
    min_leaf: int,
    seed: int,
) -> list[float]:
    """Query-grouped out-of-fold predictions; never scores a seen query."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold

    X = np.asarray([[float(row[name]) for name in feature_names] for row in features])
    y = np.asarray(target, dtype=float)
    pred = np.zeros(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(folds, len(set(groups))))
    group_to_int = {group: i for i, group in enumerate(sorted(set(groups)))}
    group_array = np.asarray([group_to_int[group] for group in groups], dtype=np.uint32)
    for train_idx, held_idx in splitter.split(X, y, group_array):
        model = RandomForestRegressor(
            n_estimators=trees,
            min_samples_leaf=min_leaf,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        pred[held_idx] = model.predict(X[held_idx])
    return pred.tolist()


def _scaled_numeric_matrix(features: list[dict], names: tuple[str, ...]):
    """Sparse, roughly [0,1]-scaled numeric features for linear text models."""
    from scipy.sparse import csr_matrix

    scales = {
        "semantic_rr": 60.0,
        "fusion_rr": 60.0,
        "graph_rr": 60.0,
        "best_rr": 60.0,
        "n_sources": 1.0 / 3.0,
        "snippet_len_words": 1.0 / 300.0,
        "domain_count": 0.1,
        "ef_count": 0.1,
        "support_count": 0.1,
        "entity_count": 0.1,
    }
    values = []
    for row in features:
        values.append([
            min(1.0, max(0.0, float(row[name]) * scales.get(name, 1.0)))
            for name in names
        ])
    return csr_matrix(np.asarray(values, dtype=float))


def _oof_ridge(
    pair_texts: list[str],
    numeric_matrix,
    target: list[float],
    groups: list[str],
    folds: int,
    alpha: float,
    min_df: int,
    max_features: int,
) -> list[float]:
    """Sparse text+numeric Ridge with fold-local TF-IDF and grouped OOF."""
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    y = np.asarray(target, dtype=float)
    pred = np.zeros(len(y), dtype=float)
    splitter = GroupKFold(n_splits=min(folds, len(set(groups))))
    group_array = np.asarray(groups)
    text_array = np.asarray(pair_texts, dtype=object)
    for train_idx, held_idx in splitter.split(numeric_matrix, y, group_array):
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=min_df, max_features=max_features,
            sublinear_tf=True,
        )
        train_text = vectorizer.fit_transform(text_array[train_idx].tolist())
        held_text = vectorizer.transform(text_array[held_idx].tolist())
        train_matrix = hstack(
            [train_text, numeric_matrix[train_idx]], format="csr")
        held_matrix = hstack(
            [held_text, numeric_matrix[held_idx]], format="csr")
        model = Ridge(alpha=alpha)
        model.fit(train_matrix, y[train_idx])
        pred[held_idx] = model.predict(held_matrix)
    return pred.tolist()


def _oof_lambdamart(
    features: list[dict],
    feature_names: tuple[str, ...],
    target: list[float],
    groups: list[str],
    folds: int,
    estimators: int,
    learning_rate: float,
    max_depth: int,
    pairs_per_sample: int,
    seed: int,
) -> list[float]:
    """Official XGBoost rank:ndcg LambdaMART with query-grouped OOF."""
    from sklearn.model_selection import GroupKFold

    X = np.asarray([[float(row[name]) for name in feature_names] for row in features])
    # XGBoost's NDCG objective expects non-negative relevance grades.  Preserve
    # the current 1--7 rubric order as seven integer levels 0--6.
    y = np.asarray([max(0, min(6, int(round(value)) - 1)) for value in target])
    pred = np.zeros(len(y), dtype=float)
    group_to_int = {group: i for i, group in enumerate(sorted(set(groups)))}
    group_array = np.asarray(
        [group_to_int[group] for group in groups], dtype=np.uint32)
    splitter = GroupKFold(n_splits=min(folds, len(set(groups))))
    for train_idx, held_idx in splitter.split(X, y, group_array):
        pred[held_idx] = _fit_lambdamart_fold(
            X, y, group_array, train_idx, held_idx, estimators, learning_rate,
            max_depth, pairs_per_sample, seed)
    return pred.tolist()


def _fit_lambdamart_fold(
    X, y, group_array, train_idx, held_idx, estimators: int,
    learning_rate: float, max_depth: int, pairs_per_sample: int, seed: int,
):
    """Canonical one-fold XGBoost fit shared by OOF and nested evaluators."""
    from xgboost import XGBRanker
    order = np.argsort(group_array[train_idx], kind="stable")
    tr = train_idx[order]
    model = XGBRanker(
        objective="rank:ndcg", tree_method="hist", n_estimators=estimators,
        learning_rate=learning_rate, max_depth=max_depth, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.5,
        lambdarank_pair_method="mean",
        lambdarank_num_pair_per_sample=pairs_per_sample,
        random_state=seed, n_jobs=-1,
    )
    model.fit(X[tr], y[tr], qid=group_array[tr], verbose=False)
    return model.predict(X[held_idx])


def _route_order(rows: list[dict], rank_field: str, k: int) -> list[int]:
    ids = [i for i, r in enumerate(rows) if int(r.get(rank_field) or 0) > 0]
    return sorted(ids, key=lambda i: (int(rows[i][rank_field]), rows[i]["comment_id"]))[:k]


def _metrics(
    rows: list[dict],
    selected: list[int],
    vectors,
    k: int,
    relevance_gate: float,
    usefulness_gate: float,
    safety_gate: float,
) -> dict:
    picked = [rows[i] for i in selected[:k]]
    utilities = [float(r["utility"]) for r in picked]
    return {
        "set_size": len(picked),
        "utility_mean": sum(utilities) / len(utilities) if utilities else 0.0,
        "utility_min": min(utilities) if utilities else 0.0,
        "utility_ndcg_at_k": graded_ndcg_at(
            [rows[i]["card_id"] for i in selected],
            {r["card_id"]: float(r["utility"]) for r in rows},
            k,
        ),
        "ild_semantic": _ild(selected[:k], vectors),
        "reasonable_share": (
            sum(_reasonable(r, relevance_gate, usefulness_gate, safety_gate) for r in picked)
            / len(picked) if picked else 0.0
        ),
        "graph_only_share": sum(int(r["graph_only"]) for r in picked) / len(picked) if picked else 0.0,
        "community_gold_share": (
            sum(int(r["community_gold"]) for r in picked) / len(picked) if picked else 0.0
        ),
    }


def _paired_query_comparison(
    per_query: list[dict], arm_a: str, arm_b: str, metric: str,
) -> dict:
    values = defaultdict(dict)
    for row in per_query:
        if row["arm"] in {arm_a, arm_b}:
            values[row["query_id"]][row["arm"]] = float(row[metric])
    deltas = [row[arm_a] - row[arm_b] for row in values.values()
              if arm_a in row and arm_b in row]
    lo, hi = bootstrap_ci(deltas, n_boot=2000, seed=20260710)
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "metric": metric,
        "queries": len(deltas),
        "mean_delta_a_minus_b": round(st.mean(deltas), 6) if deltas else None,
        "bootstrap_95_ci": [round(lo, 6), round(hi, 6)] if lo is not None else None,
        "positive_query_share": round(sum(x > 0 for x in deltas) / len(deltas), 6)
        if deltas else None,
    }


def main() -> None:
    rv = config.load().get("reranker_validation", {})
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-db", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--graph-features", type=Path, default=None,
        help="optional validation-only observed path feature CSV; enables M3",
    )
    ap.add_argument("--k", type=int, default=config.params("retrieval", "k", default=8))
    ap.add_argument("--lambdas", default=",".join(str(x) for x in rv.get(
        "diversity_lambdas", [0.0, 0.1, 0.2, 0.3, 0.4])))
    ap.add_argument("--relevance-gate", type=float, default=rv.get("relevance_gate", 3.0))
    ap.add_argument("--usefulness-gate", type=float, default=rv.get("usefulness_gate", 3.0))
    ap.add_argument("--safety-gate", type=float, default=rv.get("safety_gate", 4.0))
    ap.add_argument("--max-utility-ndcg-drop", type=float,
                    default=rv.get("max_utility_ndcg_drop", 0.005),
                    help="Pareto rule: maximize ILD subject to this drop from lambda=0.")
    ap.add_argument("--max-reasonable-share-drop", type=float,
                    default=rv.get("max_reasonable_share_drop", 0.005),
                    help="anti-pseudo-diversity constraint relative to lambda=0")
    ap.add_argument("--oof-simulated-user", action="store_true",
                    help="train query-grouped OOF proxy heads from existing LLM labels")
    ap.add_argument("--proxy-model", choices=("ridge", "rf", "lambdamart"), default="ridge",
                    help="ridge uses query+evidence text; rf diagnoses rank/facet features only")
    args = ap.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    cards = _load_cards(args.study_db, args.heldout)
    observed_graph = _load_observed_graph_features(args.graph_features)
    by_q: dict[str, list[dict]] = defaultdict(list)
    for row in cards:
        by_q[row["query_id"]].append(row)

    # Fit once for a stable corpus-level representation; rows remain query-local
    # during selection and evaluation.
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [r["snippet"] for r in cards]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000)
    all_vectors = tfidf.fit_transform(texts)
    global_pos = {r["card_id"]: i for i, r in enumerate(cards)}

    prediction_by_card: dict[str, dict[str, float]] = {}
    model_protocol = None
    if args.oof_simulated_user:
        features = _feature_rows(cards, all_vectors, tfidf, observed_graph)
        groups = [row["query_id"] for row in cards]
        folds = int(rv.get("group_folds", 5))
        target_values = {
            "utility_text": [float(r["utility"]) for r in cards],
            "utility_graph": [float(r["utility"]) for r in cards],
            "relevance_graph": [float(r["label_relevance"]) for r in cards],
            "usefulness_graph": [float(r["label_usefulness"]) for r in cards],
            "safety_graph": [float(r["label_safety"]) for r in cards],
        }
        if observed_graph:
            target_values["utility_rich_graph"] = [float(r["utility"]) for r in cards]
        if args.proxy_model == "rf":
            settings = {
                "folds": folds,
                "trees": int(rv.get("rf_trees", 250)),
                "min_leaf": int(rv.get("rf_min_leaf", 2)),
                "seed": int(rv.get("seed", 13)),
            }
            feature_names = {
                "utility_text": TEXT_FEATURES,
                "utility_graph": GRAPH_FEATURES,
                "relevance_graph": GRAPH_FEATURES,
                "usefulness_graph": GRAPH_FEATURES,
                "safety_graph": GRAPH_FEATURES,
            }
            if observed_graph:
                feature_names["utility_rich_graph"] = RICH_GRAPH_FEATURES
            predictions = {
                name: _oof_rf(features, feature_names[name], target, groups, **settings)
                for name, target in target_values.items()
            }
            model_name = "RandomForestRegressor rank/facet diagnostic proxy"
            representation = "numeric retrieval/facet features"
            prediction_protocol = "query-grouped out-of-fold"
        elif args.proxy_model == "lambdamart":
            settings = {
                "folds": folds,
                "estimators": int(rv.get("lambdamart_estimators", 200)),
                "learning_rate": float(rv.get("lambdamart_learning_rate", 0.05)),
                "max_depth": int(rv.get("lambdamart_max_depth", 3)),
                "pairs_per_sample": int(rv.get("lambdamart_pairs_per_sample", 8)),
                "seed": int(rv.get("seed", 13)),
            }
            feature_names = {
                "utility_text": TEXT_FEATURES,
                "utility_graph": GRAPH_FEATURES,
                "relevance_graph": GRAPH_FEATURES,
                "usefulness_graph": GRAPH_FEATURES,
                "safety_graph": GRAPH_FEATURES,
            }
            if observed_graph:
                feature_names["utility_rich_graph"] = RICH_GRAPH_FEATURES
            predictions = {
                name: _oof_lambdamart(
                    features, feature_names[name], target, groups, **settings)
                for name, target in target_values.items()
            }
            model_name = "XGBoost LambdaMART rank:ndcg validation proxy"
            representation = "numeric retrieval/facet features; 7-level graded targets"
            prediction_protocol = "query-grouped out-of-fold; mean pair sampling"
        else:
            pair_texts = [
                f"{row['query_text']} [SEP] {row['snippet']}" for row in cards
            ]
            text_matrix = _scaled_numeric_matrix(features, TEXT_FEATURES)
            graph_matrix = _scaled_numeric_matrix(features, GRAPH_FEATURES)
            matrices = {
                "utility_text": text_matrix,
                "utility_graph": graph_matrix,
                "relevance_graph": graph_matrix,
                "usefulness_graph": graph_matrix,
                "safety_graph": graph_matrix,
            }
            if observed_graph:
                matrices["utility_rich_graph"] = _scaled_numeric_matrix(
                    features, RICH_GRAPH_FEATURES)
            settings = {
                "folds": folds,
                "alpha": float(rv.get("ridge_alpha", 10.0)),
                "min_df": int(rv.get("tfidf_min_df", 2)),
                "max_features": int(rv.get("tfidf_max_features", 30_000)),
            }
            predictions = {
                name: _oof_ridge(pair_texts, matrices[name], target, groups, **settings)
                for name, target in target_values.items()
            }
            model_name = "Ridge query-evidence text validation proxy"
            representation = "word 1-2gram TF-IDF + scaled retrieval/facet features"
            prediction_protocol = (
                "query-grouped out-of-fold; text vectorizer fit inside each training fold")
        for i, card in enumerate(cards):
            prediction_by_card[card["card_id"]] = {
                name: float(values[i]) for name, values in predictions.items()
            }
        model_protocol = {
            "model": model_name,
            "proxy_model": args.proxy_model,
            "representation": representation,
            "prediction": prediction_protocol,
            **settings,
            "text_features": list(TEXT_FEATURES),
            "text_plus_graph_features": list(GRAPH_FEATURES),
            "m1_non_graph_features": list(TEXT_FEATURES),
            "m2_coarse_graph_features": list(GRAPH_FEATURES),
            "m3_observed_graph_features": (
                list(RICH_GRAPH_FEATURES) if observed_graph else None),
            "targets": list(target_values),
            "label_role": "LLM simulated-user proxy; not human ground truth",
        }

    per_query: list[dict] = []
    arms = ["semantic", "graph", "union_best_rank", "llm_utility_gate"]
    arms += [f"llm_utility_mmr_lambda_{lam:g}" for lam in lambdas if lam > 0]
    if args.oof_simulated_user:
        arms += [
            "oof_relevance_only_graph_features",
            "oof_composite_utility_text_features",
            "oof_composite_utility_graph_features",
            "oof_composite_utility_predicted_gate",
        ]
        if observed_graph:
            arms.append("oof_composite_utility_rich_graph_features")
        arms += [f"oof_composite_utility_mmr_lambda_{lam:g}"
                 for lam in lambdas if lam > 0]
    for qid in sorted(by_q):
        rows = by_q[qid]
        idx = [global_pos[r["card_id"]] for r in rows]
        vectors = all_vectors[idx]
        reasonable = [
            i for i, r in enumerate(rows)
            if _reasonable(r, args.relevance_gate, args.usefulness_gate, args.safety_gate)
        ]
        selections = {
            "semantic": _route_order(rows, "semantic_rank", args.k),
            "graph": _route_order(rows, "graph_rank", args.k),
            "union_best_rank": sorted(
                range(len(rows)),
                key=lambda i: (
                    min([x for x in (rows[i]["semantic_rank"], rows[i]["fusion_dense_bm25_rank"],
                                     rows[i]["graph_rank"]) if x] or [9999]),
                    rows[i]["comment_id"],
                ),
            )[:args.k],
            "llm_utility_gate": _mmr_select(rows, vectors, reasonable, args.k, 0.0),
        }
        for lam in lambdas:
            if lam > 0:
                selections[f"llm_utility_mmr_lambda_{lam:g}"] = _mmr_select(
                    rows, vectors, reasonable, args.k, lam)
        if args.oof_simulated_user:
            pred = [prediction_by_card[row["card_id"]] for row in rows]
            rel_scores = [p["relevance_graph"] for p in pred]
            util_text_scores = [p["utility_text"] for p in pred]
            util_graph_scores = [p["utility_graph"] for p in pred]
            util_rich_scores = (
                [p["utility_rich_graph"] for p in pred] if observed_graph else None)
            if args.proxy_model == "lambdamart":
                # LambdaMART scores are ordinal and uncalibrated.  Use all
                # candidates for the ranking comparison; a separate calibrated
                # gate must be learned by a regression/classification head.
                predicted_reasonable = list(range(len(rows)))
                lo, hi = min(util_graph_scores), max(util_graph_scores)
                if hi > lo:
                    util_graph_scores = [1.0 + 6.0 * (x - lo) / (hi - lo)
                                         for x in util_graph_scores]
            else:
                predicted_reasonable = [
                    i for i, p in enumerate(pred)
                    if p["relevance_graph"] >= args.relevance_gate
                    and p["usefulness_graph"] >= args.usefulness_gate
                    and p["safety_graph"] >= args.safety_gate
                ]
            selections.update({
                "oof_relevance_only_graph_features": sorted(
                    range(len(rows)), key=lambda i: (-rel_scores[i], rows[i]["comment_id"]))[:args.k],
                "oof_composite_utility_text_features": sorted(
                    range(len(rows)), key=lambda i: (-util_text_scores[i], rows[i]["comment_id"]))[:args.k],
                "oof_composite_utility_graph_features": sorted(
                    range(len(rows)), key=lambda i: (-util_graph_scores[i], rows[i]["comment_id"]))[:args.k],
                "oof_composite_utility_predicted_gate": _mmr_select(
                    rows, vectors, predicted_reasonable, args.k, 0.0, util_graph_scores),
            })
            if util_rich_scores is not None:
                selections["oof_composite_utility_rich_graph_features"] = sorted(
                    range(len(rows)),
                    key=lambda i: (-util_rich_scores[i], rows[i]["comment_id"]),
                )[:args.k]
            for lam in lambdas:
                if lam > 0:
                    selections[f"oof_composite_utility_mmr_lambda_{lam:g}"] = _mmr_select(
                        rows, vectors, predicted_reasonable, args.k, lam, util_graph_scores)
        for arm in arms:
            metrics = _metrics(
                rows, selections[arm], vectors, args.k,
                args.relevance_gate, args.usefulness_gate, args.safety_gate,
            )
            per_query.append({"query_id": qid, "arm": arm, **metrics})

    summary: list[dict] = []
    for arm in arms:
        subset = [r for r in per_query if r["arm"] == arm]
        row = {"arm": arm, "queries": len(subset)}
        for key in (
            "set_size", "utility_mean", "utility_min", "utility_ndcg_at_k",
            "ild_semantic", "reasonable_share", "graph_only_share", "community_gold_share",
        ):
            row[key] = round(sum(float(r[key]) for r in subset) / len(subset), 6)
        summary.append(row)

    baseline = next(r for r in summary if r["arm"] == "llm_utility_gate")
    eligible = [
        r for r in summary
        if r["arm"].startswith("llm_utility_mmr_lambda_")
        and baseline["utility_ndcg_at_k"] - r["utility_ndcg_at_k"]
        <= args.max_utility_ndcg_drop + 1e-12
        and baseline["reasonable_share"] - r["reasonable_share"]
        <= args.max_reasonable_share_drop + 1e-12
    ]
    recommended = max(eligible, key=lambda r: r["ild_semantic"]) if eligible else baseline

    oof_recommended = None
    simulated_user_comparison = None
    if args.oof_simulated_user:
        by_arm = {row["arm"]: row for row in summary}
        oof_base = by_arm["oof_composite_utility_predicted_gate"]
        oof_eligible = [
            row for row in summary
            if row["arm"].startswith("oof_composite_utility_mmr_lambda_")
            and oof_base["utility_ndcg_at_k"] - row["utility_ndcg_at_k"]
            <= args.max_utility_ndcg_drop + 1e-12
            and oof_base["reasonable_share"] - row["reasonable_share"]
            <= args.max_reasonable_share_drop + 1e-12
        ]
        oof_recommended = max(
            oof_eligible, key=lambda row: row["ild_semantic"]) if oof_eligible else oof_base
        gate_calibrated = args.proxy_model != "lambdamart"
        rel = by_arm["oof_relevance_only_graph_features"]
        util_text = by_arm["oof_composite_utility_text_features"]
        util_graph = by_arm["oof_composite_utility_graph_features"]
        simulated_user_comparison = {
            "composite_vs_relevance_utility_ndcg_delta": round(
                util_graph["utility_ndcg_at_k"] - rel["utility_ndcg_at_k"], 6),
            "graph_feature_utility_ndcg_delta": round(
                util_graph["utility_ndcg_at_k"] - util_text["utility_ndcg_at_k"], 6),
            "gate_calibrated": gate_calibrated,
            "mmr_recommendation_valid": gate_calibrated,
            "recommended_oof_diversity_arm": (
                oof_recommended["arm"] if gate_calibrated else None),
            "recommended_oof_diversity_metrics": (
                oof_recommended if gate_calibrated else None),
            "diagnostic_oof_diversity_arm": oof_recommended["arm"],
            "paired_query_tests": {
                "composite_vs_relevance": _paired_query_comparison(
                    per_query,
                    "oof_composite_utility_graph_features",
                    "oof_relevance_only_graph_features",
                    "utility_ndcg_at_k",
                ),
                "graph_vs_text_features": _paired_query_comparison(
                    per_query,
                    "oof_composite_utility_graph_features",
                    "oof_composite_utility_text_features",
                    "utility_ndcg_at_k",
                ),
                "recommended_mmr_vs_predicted_gate": _paired_query_comparison(
                    per_query,
                    oof_recommended["arm"],
                    "oof_composite_utility_predicted_gate",
                    "utility_ndcg_at_k",
                ),
            },
        }
        if observed_graph:
            rich = by_arm["oof_composite_utility_rich_graph_features"]
            simulated_user_comparison["m1_m2_m3"] = {
                "m1_arm": "oof_composite_utility_text_features",
                "m2_arm": "oof_composite_utility_graph_features",
                "m3_arm": "oof_composite_utility_rich_graph_features",
                "m2_minus_m1_utility_ndcg": round(
                    util_graph["utility_ndcg_at_k"] - util_text["utility_ndcg_at_k"], 6),
                "m3_minus_m2_utility_ndcg": round(
                    rich["utility_ndcg_at_k"] - util_graph["utility_ndcg_at_k"], 6),
                "m3_minus_m1_utility_ndcg": round(
                    rich["utility_ndcg_at_k"] - util_text["utility_ndcg_at_k"], 6),
                "paired_query_tests": {
                    "m2_vs_m1": _paired_query_comparison(
                        per_query, "oof_composite_utility_graph_features",
                        "oof_composite_utility_text_features", "utility_ndcg_at_k"),
                    "m3_vs_m2": _paired_query_comparison(
                        per_query, "oof_composite_utility_rich_graph_features",
                        "oof_composite_utility_graph_features", "utility_ndcg_at_k"),
                    "m3_vs_m1": _paired_query_comparison(
                        per_query, "oof_composite_utility_rich_graph_features",
                        "oof_composite_utility_text_features", "utility_ndcg_at_k"),
                },
            }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("per_query.csv", per_query), ("summary.csv", summary)):
        with (args.out_dir / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    if args.oof_simulated_user:
        prediction_rows = []
        for card in cards:
            prediction_rows.append({
                "query_id": card["query_id"],
                "comment_id": card["comment_id"],
                "card_id": card["card_id"],
                "label_utility": card["utility"],
                "label_relevance": card["label_relevance"],
                "label_usefulness": card["label_usefulness"],
                "label_safety": card["label_safety"],
                **{f"pred_{k}": round(v, 6)
                   for k, v in prediction_by_card[card["card_id"]].items()},
            })
        with (args.out_dir / "oof_predictions.csv").open(
                "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(prediction_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(prediction_rows)
    manifest = {
        "protocol": "validation quality-diversity v2",
        "test_split_used": False,
        "quality_source": "existing LLM evidence labels (diagnostic oracle)",
        "final_system_quality_source": "out-of-fold query-grouped reranker prediction",
        "diversity_source": "TF-IDF text cosine; list-level MMR",
        "k": args.k,
        "queries": len(by_q),
        "cards": len(cards),
        "gates": {
            "relevance": args.relevance_gate,
            "usefulness": args.usefulness_gate,
            "safety": args.safety_gate,
        },
        "diversity_lambdas": lambdas,
        "simulated_user_oof": model_protocol,
        "simulated_user_comparison": simulated_user_comparison,
        "observed_graph_features": {
            "path": str(args.graph_features) if args.graph_features else None,
            "enabled": bool(observed_graph),
            "rows_loaded": len(observed_graph),
            "join_key": ["query_id", "comment_id"],
        },
        "pareto_rule": {
            "maximize": "ild_semantic",
            "constraint": "utility_ndcg_at_k drop from gated lambda=0",
            "max_drop": args.max_utility_ndcg_drop,
            "max_reasonable_share_drop": args.max_reasonable_share_drop,
            "recommended_validation_arm": recommended["arm"],
            "recommended_oof_arm": (
                oof_recommended["arm"]
                if oof_recommended and args.proxy_model != "lambdamart" else None),
            "diagnostic_oof_arm": oof_recommended["arm"] if oof_recommended else None,
        },
        "summary": summary,
        "interpretation_guard": (
            "LLM utility may supervise quality, but ILD is list-relative and must not be copied into "
            "a pointwise target. Tune lambda on validation and calibrate quality/gates with humans. "
            "LambdaMART scores are query-relative ranking scores, so their raw values are not a "
            "deployable safety/usefulness threshold."
        ),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
