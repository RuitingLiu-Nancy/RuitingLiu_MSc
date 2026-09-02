#!/usr/bin/env python3
"""Audit Dense Top-8 mismatch and matched one-swap rescue routes.

The audit is development-only and deliberately fail-closed:

* the frozen SBERT D8 baseline is never re-selected;
* candidate pools are constructed without utility, mismatch, or Oracle labels;
* incomplete BM25/Graph-practical judgments remain missing and are exported as
  residual manifests rather than imputed;
* the realised policy is the report89 Direct Huber/NO-OP/nested-OOF procedure,
  using only source-common features in its primary specification;
* frozen test paths and external model calls are rejected.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as _xgboost  # noqa: F401  # keep native import order

try:
    import configuration as project_config
    from evaluation.judgment_completeness import (
        complete_utility_v2_rows,
    )
    from evaluation.statistics import bootstrap_ci
    from evaluation.safety_filter import DrugFilter
    from utility_scoring.learned_diffusion import reranker_validation as canonical
    from candidate_pool.analyze_strict_sbert_graph_oracle import (
        _read_jsonl,
        _reject_test,
        _sha256,
        _write_json,
        _write_jsonl,
    )
    from evidence_selection.run_strict_native_graph_conservative_policy import (
        _cosine,
        _evaluate_fold_variant,
        _pairwise_set_similarity,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import configuration as project_config
    from evaluation.judgment_completeness import (
        complete_utility_v2_rows,
    )
    from evaluation.statistics import bootstrap_ci
    from evaluation.safety_filter import DrugFilter
    from utility_scoring.learned_diffusion import reranker_validation as canonical
    from candidate_pool.analyze_strict_sbert_graph_oracle import (
        _read_jsonl,
        _reject_test,
        _sha256,
        _write_json,
        _write_jsonl,
    )
    from evidence_selection.run_strict_native_graph_conservative_policy import (
        _cosine,
        _evaluate_fold_variant,
        _pairwise_set_similarity,
    )


POOL_DEEP = "Deep4"
POOL_BM25_PRACTICAL = "BM25-practical4"
POOL_GRAPH_PRACTICAL = "Graph-practical4"
POOL_BM25_EXCLUSIVE = "BM25-exclusive4"
POOL_GRAPH_EXCLUSIVE = "Graph-exclusive4"
POOL_ORDER = (
    POOL_DEEP,
    POOL_BM25_PRACTICAL,
    POOL_GRAPH_PRACTICAL,
    POOL_BM25_EXCLUSIVE,
    POOL_GRAPH_EXCLUSIVE,
)

MISMATCH_A = "A_semantic_context"
MISMATCH_B = "B_support_quality"
MISMATCH_C = "C_safety"

DIMENSIONS = {
    "R": "relevance",
    "H": "usefulness",
    "A": "actionability",
    "N": "novelty",
    "E": "resonance",
    "S": "safety",
}
UTILITY_WEIGHTS = {
    "R": 0.25,
    "H": 0.30,
    "A": 0.15,
    "N": 0.10,
    "E": 0.10,
    "S": 0.10,
}

STATIC_PREDICTOR_FEATURES = (
    "query_candidate_sbert_similarity",
    "candidate_dense_percentile",
    "candidate_dense_rank_missing",
    "comment_length_log",
    "idf_weighted_lexical_overlap",
    "candidate_to_d8_max_similarity",
    "candidate_to_d8_mean_similarity",
    "candidate_novelty_relative_to_d8",
)

SOURCE_BLIND_FEATURES = (
    "query_candidate_sbert_similarity",
    "candidate_dense_percentile",
    "candidate_dense_rank_missing",
    "comment_length_log",
    "idf_weighted_lexical_overlap",
    "candidate_to_d8_max_similarity",
    "candidate_to_d8_mean_similarity",
    "candidate_novelty_relative_to_d8",
    "candidate_to_replaced_item_similarity",
    "oof_predicted_utility_difference",
    "query_similarity_difference",
    "comment_length_difference",
    "idf_overlap_difference",
    "d8_similarity_dispersion",
    "d8_redundancy",
)

ROUTE_AWARE_FEATURES = (
    *SOURCE_BLIND_FEATURES,
    "bm25_rank_norm",
    "bm25_score_relative",
    "bm25_route_missing",
    "graph_ppr_rank_norm",
    "graph_ppr_score_relative",
    "graph_route_missing",
)

FORBIDDEN_INFERENCE_FEATURES = (
    "gold",
    "raw_reward",
    "oracle",
    "opportunity",
    "mismatch_label",
    "source_identity",
    "route_identity_bonus",
    "quota",
)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["query_id", "source_pool", "analysis_scope"]
    fieldnames = [
        *[name for name in preferred if name in fieldnames],
        *[name for name in fieldnames if name not in preferred],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _bootstrap_summary(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> dict:
    if not values:
        return {
            "n_queries": 0,
            "mean": None,
            "query_bootstrap_95ci": [None, None],
        }
    lo, hi = bootstrap_ci(values, n_boot=samples, seed=seed)
    return {
        "n_queries": len(values),
        "mean": statistics.fmean(values),
        "query_bootstrap_95ci": [lo, hi],
    }


def _wtl(values: Iterable[float], atol: float = 1e-12) -> dict:
    values = list(values)
    return {
        "wins": sum(value > atol for value in values),
        "ties": sum(abs(value) <= atol for value in values),
        "losses": sum(value < -atol for value in values),
    }


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"(?u)\b\w\w+\b", str(text).lower())
        if token not in ENGLISH_STOP_WORDS
    ]


def _build_idf(corpus_texts: Iterable[str]) -> tuple[dict[str, float], dict]:
    documents = [set(_tokenize(text)) for text in corpus_texts]
    df = Counter(token for document in documents for token in document)
    count = len(documents)
    idf = {
        token: math.log((count + 1.0) / (frequency + 1.0)) + 1.0
        for token, frequency in df.items()
    }
    return idf, {
        "documents": count,
        "vocabulary": len(idf),
        "formula": "log((N+1)/(df+1))+1",
        "token_pattern": r"(?u)\b\w\w+\b",
        "stopwords": "sklearn ENGLISH_STOP_WORDS",
        "stemming": False,
    }


def _lexical_diagnostics(
    query_text: str,
    candidate_text: str,
    idf: dict[str, float],
) -> dict:
    query_tokens = set(_tokenize(query_text))
    candidate_tokens = set(_tokenize(candidate_text))
    shared = query_tokens & candidate_tokens
    denominator = sum(idf.get(token, 1.0) for token in query_tokens)
    overlap = (
        sum(idf.get(token, 1.0) for token in shared) / denominator
        if denominator
        else 0.0
    )
    rare = sorted(
        shared,
        key=lambda token: (-idf.get(token, 1.0), token),
    )
    rare_threshold = (
        float(np.quantile([idf.get(token, 1.0) for token in query_tokens], 0.75))
        if query_tokens
        else math.inf
    )
    recovered = [
        token for token in rare if idf.get(token, 1.0) >= rare_threshold
    ]
    return {
        "idf_weighted_lexical_overlap": overlap,
        "shared_token_count": len(shared),
        "rare_query_token_recovery_count": len(recovered),
        "rare_query_tokens_recovered": recovered[:20],
    }


def _minmax_map(rows: list[dict], value_key: str) -> dict[str, float]:
    values = [float(row[value_key]) for row in rows]
    low, high = min(values), max(values)
    if high <= low:
        return {str(row["comment_id"]): 0.0 for row in rows}
    return {
        str(row["comment_id"]): (float(row[value_key]) - low) / (high - low)
        for row in rows
    }


def _normalise_structured_run(path: Path, qids: set[str]) -> dict[str, list[dict]]:
    output = {}
    for row in _read_jsonl(path):
        query_id = str(row["query_id"])
        if query_id not in qids:
            continue
        output[query_id] = [
            {
                "query_id": query_id,
                "comment_id": str(candidate_id),
                "rank": rank,
                "score": float(score),
            }
            for rank, (candidate_id, score) in enumerate(
                zip(row["retrieved_titles"], row["retrieved_scores"], strict=True),
                start=1,
            )
        ]
    if set(output) != qids:
        raise ValueError(f"{path}: query identity mismatch")
    return output


def _normalise_flat_run(path: Path, qids: set[str]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = defaultdict(list)
    for row in _read_jsonl(path):
        query_id = str(row["query_id"])
        if query_id in qids:
            output[query_id].append({
                "query_id": query_id,
                "comment_id": str(row["comment_id"]),
                "rank": int(row["rank"]),
                "score": float(row["score"]),
            })
    output = {
        query_id: sorted(rows, key=lambda row: (row["rank"], row["comment_id"]))
        for query_id, rows in output.items()
    }
    if set(output) != qids:
        raise ValueError(f"{path}: query identity mismatch")
    if any(
        len(rows) != 100
        or [row["rank"] for row in rows] != list(range(1, 101))
        or len({row["comment_id"] for row in rows}) != 100
        for rows in output.values()
    ):
        raise ValueError(f"{path}: BM25 Top-100 integrity failed")
    return output


def _load_embeddings(
    *,
    corpus_path: Path,
    corpus_embeddings_path: Path,
    queries_path: Path,
    query_embeddings_path: Path,
    qids: set[str],
) -> tuple[dict, dict, dict, dict, np.ndarray, list[str]]:
    corpus_rows = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus_ids = [str(row["title"]) for row in corpus_rows]
    corpus_text = {str(row["title"]): str(row["text"]) for row in corpus_rows}
    corpus_index = {candidate_id: index for index, candidate_id in enumerate(corpus_ids)}
    if len(corpus_index) != len(corpus_ids):
        raise ValueError("duplicate corpus candidate IDs")
    corpus_embeddings = np.load(corpus_embeddings_path, mmap_mode="r")
    if corpus_embeddings.shape[0] != len(corpus_ids):
        raise ValueError("corpus embedding order mismatch")

    query_rows = json.loads(queries_path.read_text(encoding="utf-8"))
    query_text = {str(row["id"]): str(row["question"]) for row in query_rows}
    query_ids = sorted(query_text)
    query_index = {query_id: index for index, query_id in enumerate(query_ids)}
    query_embeddings = np.load(query_embeddings_path, mmap_mode="r")
    if query_embeddings.shape[0] != len(query_ids) or not qids <= set(query_ids):
        raise ValueError("query embedding order/identity mismatch")

    candidate_vectors = {
        candidate_id: np.asarray(
            corpus_embeddings[index],
            dtype=np.float32,
        )
        for candidate_id, index in corpus_index.items()
    }
    query_vectors = {
        query_id: np.asarray(
            query_embeddings[query_index[query_id]],
            dtype=np.float32,
        )
        for query_id in qids
    }
    return (
        candidate_vectors,
        query_vectors,
        corpus_text,
        query_text,
        corpus_embeddings,
        corpus_ids,
    )


def _dimensions(row: dict) -> dict[str, float]:
    return {
        short: float(row[f"label_{name}"])
        for short, name in DIMENSIONS.items()
    } | {"U": float(row["utility"])}


def _linear_utility(row: dict) -> float:
    return sum(
        UTILITY_WEIGHTS[short] * float(row[f"label_{DIMENSIONS[short]}"])
        for short in UTILITY_WEIGHTS
    )


def _safety_gate_residual(row: dict) -> float:
    return float(row["utility"]) - _linear_utility(row)


def _mismatch_type(row: dict, mismatch_cfg: dict) -> str | None:
    relevance = float(row["label_relevance"])
    usefulness = float(row["label_usefulness"])
    actionability = float(row["label_actionability"])
    safety = float(row["label_safety"])
    utility = float(row["utility"])
    semantic = mismatch_cfg["semantic_context"]
    support = mismatch_cfg["support_quality"]
    safety_cfg = mismatch_cfg["safety"]
    if safety <= float(safety_cfg["safety_max"]):
        return MISMATCH_C
    if (
        safety > float(semantic["safety_min_exclusive"])
        and relevance <= float(semantic["relevance_max"])
    ):
        return MISMATCH_A
    if (
        safety > float(support["safety_min_exclusive"])
        and relevance >= float(support["relevance_min"])
        and utility < float(support["utility_max_exclusive"])
        and (
            usefulness <= float(support["usefulness_max"])
            or actionability <= float(support["actionability_max"])
        )
    ):
        return MISMATCH_B
    return None


def _round_robin_graph_pool(
    query_id: str,
    routes: dict[str, dict[str, list[dict]]],
    excluded: set[str],
    budget: int,
) -> list[dict]:
    names = sorted(routes)
    indices = {name: 0 for name in names}
    chosen: list[dict] = []
    seen: set[str] = set()
    while len(chosen) < budget:
        progressed = False
        for name in names:
            rows = routes[name][query_id]
            while indices[name] < len(rows):
                row = rows[indices[name]]
                indices[name] += 1
                candidate_id = str(row["comment_id"])
                if candidate_id in excluded or candidate_id in seen:
                    continue
                seen.add(candidate_id)
                chosen.append({
                    **row,
                    "first_route": name,
                })
                progressed = True
                break
            if len(chosen) >= budget:
                break
        if not progressed:
            break
    return chosen


def _source_route_fields(
    *,
    query_id: str,
    candidate_id: str,
    bm25: dict[str, list[dict]],
    graph_routes: dict[str, dict[str, list[dict]]],
) -> dict:
    bm25_by_id = {
        str(row["comment_id"]): row for row in bm25[query_id]
    }
    bm25_relative = _minmax_map(bm25[query_id], "score")
    graph_rank: dict[str, int] = {}
    graph_score: dict[str, float] = {}
    graph_relative: dict[str, float] = {}
    for route_name, route_run in graph_routes.items():
        route_rows = route_run[query_id]
        route_by_id = {
            str(row["comment_id"]): row for row in route_rows
        }
        relative = _minmax_map(route_rows, "score")
        if candidate_id in route_by_id:
            graph_rank[route_name] = int(route_by_id[candidate_id]["rank"])
            graph_score[route_name] = float(route_by_id[candidate_id]["score"])
            graph_relative[route_name] = float(relative[candidate_id])
    bm25_row = bm25_by_id.get(candidate_id)
    return {
        "bm25_rank": int(bm25_row["rank"]) if bm25_row else None,
        "bm25_score": float(bm25_row["score"]) if bm25_row else None,
        "bm25_score_relative": (
            float(bm25_relative[candidate_id]) if bm25_row else 0.0
        ),
        "graph_pre_fallback_rank": graph_rank,
        "native_graph_score": graph_score,
        "graph_score_relative": graph_relative,
        "graph_routes": sorted(graph_rank),
    }


def _build_candidate_pools(
    *,
    qids: set[str],
    feature_rows: list[dict],
    dense: dict[str, list[dict]],
    bm25: dict[str, list[dict]],
    graph_routes: dict[str, dict[str, list[dict]]],
    strict_graph_rows: list[dict],
    registry: dict[tuple[str, str], dict],
    candidate_vectors: dict[str, np.ndarray],
    query_vectors: dict[str, np.ndarray],
    corpus_matrix: np.ndarray,
    corpus_text: dict[str, str],
    query_text: dict[str, str],
    idf: dict[str, float],
    budget: int,
) -> tuple[
    dict[str, dict[str, list[dict]]],
    list[dict],
    list[dict],
    dict[str, dict[str, float]],
]:
    by_source: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in feature_rows:
        by_source[str(row["query_id"])][str(row["candidate_source"])].append(row)
    for query_id in qids:
        for source in by_source[query_id]:
            by_source[query_id][source].sort(
                key=lambda row: (int(row["source_rank"]), str(row["comment_id"]))
            )

    strict_by_query: dict[str, list[dict]] = defaultdict(list)
    for row in strict_graph_rows:
        strict_by_query[str(row["query_id"])].append(row)
    for query_id in qids:
        strict_by_query[query_id].sort(
            key=lambda row: (
                int(row["reported_source_rank"]),
                str(row["candidate_id"]),
            )
        )

    pool_ids: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for query_id in sorted(qids):
        d8 = [
            str(row["comment_id"])
            for row in by_source[query_id]["dense_top8"]
        ]
        depth = by_source[query_id]["dense_depth_control"]
        if len(d8) != 8 or len(depth) != budget:
            raise ValueError(f"{query_id}: frozen D8/Depth4 budget mismatch")
        dense100 = {str(row["comment_id"]) for row in dense[query_id][:100]}
        bm25_practical = [
            row for row in bm25[query_id]
            if str(row["comment_id"]) not in set(d8)
        ][:budget]
        bm25_exclusive = [
            row for row in bm25[query_id]
            if str(row["comment_id"]) not in dense100
        ][:budget]
        graph_practical = _round_robin_graph_pool(
            query_id,
            graph_routes,
            set(d8),
            budget,
        )
        graph_exclusive = strict_by_query[query_id]
        if any(len(rows) != budget for rows in (
            depth,
            bm25_practical,
            graph_practical,
            bm25_exclusive,
            graph_exclusive,
        )):
            raise ValueError(f"{query_id}: one or more rescue pools are incomplete")
        pool_ids[query_id] = {
            POOL_DEEP: [
                {
                    "comment_id": str(row["comment_id"]),
                    "source_rank": int(row["source_rank"]),
                }
                for row in depth
            ],
            POOL_BM25_PRACTICAL: [
                {
                    "comment_id": str(row["comment_id"]),
                    "source_rank": int(row["rank"]),
                }
                for row in bm25_practical
            ],
            POOL_GRAPH_PRACTICAL: [
                {
                    "comment_id": str(row["comment_id"]),
                    "source_rank": rank,
                    "first_route": str(row["first_route"]),
                }
                for rank, row in enumerate(graph_practical, start=1)
            ],
            POOL_BM25_EXCLUSIVE: [
                {
                    "comment_id": str(row["comment_id"]),
                    "source_rank": int(row["rank"]),
                }
                for row in bm25_exclusive
            ],
            POOL_GRAPH_EXCLUSIVE: [
                {
                    "comment_id": str(row["candidate_id"]),
                    "source_rank": int(row["reported_source_rank"]),
                    "strict_provenance": row,
                }
                for row in graph_exclusive
            ],
        }

    score_context: dict[str, dict[str, float]] = {}
    percentile_context: dict[str, dict[str, float]] = {}
    for query_id in sorted(qids):
        scores = np.asarray(
            corpus_matrix @ query_vectors[query_id],
            dtype=np.float32,
        )
        sorted_scores = np.sort(scores)
        candidate_ids = {
            str(row["comment_id"])
            for row in by_source[query_id]["dense_top8"]
        }
        for pool in POOL_ORDER:
            candidate_ids.update(
                str(row["comment_id"]) for row in pool_ids[query_id][pool]
            )
        score_context[query_id] = {
            candidate_id: _cosine(
                query_vectors[query_id],
                candidate_vectors[candidate_id],
            )
            for candidate_id in candidate_ids
        }
        percentile_context[query_id] = {
            candidate_id: float(
                np.searchsorted(
                    sorted_scores,
                    score_context[query_id][candidate_id],
                    side="right",
                )
                / len(sorted_scores)
            )
            for candidate_id in candidate_ids
        }

    drug_filter = DrugFilter()
    dense_mismatch_rows: list[dict] = []
    pool_rows: list[dict] = []
    dense_rank_maps = {
        query_id: {
            str(row["comment_id"]): int(row["rank"])
            for row in dense[query_id]
        }
        for query_id in qids
    }
    for query_id in sorted(qids):
        d8_rows = by_source[query_id]["dense_top8"]
        d8_ids = [str(row["comment_id"]) for row in d8_rows]
        d8_vectors = [candidate_vectors[candidate_id] for candidate_id in d8_ids]
        d8_internal_mean, d8_internal_max = _pairwise_set_similarity(d8_vectors)
        for row in d8_rows:
            candidate_id = str(row["comment_id"])
            judgment = registry[(query_id, candidate_id)]
            dense_rank = dense_rank_maps[query_id][candidate_id]
            dense_mismatch_rows.append({
                "query_id": query_id,
                "candidate_id": candidate_id,
                "dense_rank": dense_rank,
                "cosine_similarity": score_context[query_id][candidate_id],
                "within_query_similarity_percentile": (
                    percentile_context[query_id][candidate_id]
                ),
                **_dimensions(judgment),
                "safety_gate_status": (
                    "gated" if float(judgment["label_safety"]) <= 2 else "not_gated"
                ),
                "mismatch_type": None,
            })

        for pool in POOL_ORDER:
            for item in pool_ids[query_id][pool]:
                candidate_id = str(item["comment_id"])
                if candidate_id in d8_ids:
                    raise ValueError(f"{query_id}/{pool}: candidate overlaps D8")
                vector = candidate_vectors[candidate_id]
                d8_similarities = [
                    _cosine(vector, d8_vector) for d8_vector in d8_vectors
                ]
                lexical = _lexical_diagnostics(
                    query_text[query_id],
                    corpus_text[candidate_id],
                    idf,
                )
                route = _source_route_fields(
                    query_id=query_id,
                    candidate_id=candidate_id,
                    bm25=bm25,
                    graph_routes=graph_routes,
                )
                dense_rank = dense_rank_maps[query_id].get(candidate_id)
                judgment = registry.get((query_id, candidate_id))
                query_medication = drug_filter.match(query_text[query_id])
                candidate_medication = drug_filter.match(corpus_text[candidate_id])
                provenance = item.get("strict_provenance")
                if pool == POOL_GRAPH_EXCLUSIVE:
                    if not (
                        provenance
                        and bool(provenance["native_graph"])
                        and not bool(provenance["fallback_used"])
                        and not bool(provenance["callback_used"])
                        and not bool(provenance["padding_used"])
                    ):
                        raise ValueError("strict Graph-exclusive provenance failed")
                if pool == POOL_GRAPH_PRACTICAL:
                    if not route["graph_pre_fallback_rank"]:
                        raise ValueError("Graph-practical candidate lacks graph route")
                    if not all(
                        math.isfinite(score) and score > 0
                        for score in route["native_graph_score"].values()
                    ):
                        raise ValueError("Graph-practical PPR score is invalid")
                static = {
                    "query_candidate_sbert_similarity": (
                        score_context[query_id][candidate_id]
                    ),
                    "candidate_dense_percentile": (
                        (101 - dense_rank) / 100.0 if dense_rank is not None else 0.0
                    ),
                    "candidate_dense_rank_missing": float(dense_rank is None),
                    "comment_length_log": math.log1p(
                        len(_tokenize(corpus_text[candidate_id]))
                    ),
                    "idf_weighted_lexical_overlap": lexical[
                        "idf_weighted_lexical_overlap"
                    ],
                    "candidate_to_d8_max_similarity": max(d8_similarities),
                    "candidate_to_d8_mean_similarity": statistics.fmean(
                        d8_similarities
                    ),
                    "candidate_novelty_relative_to_d8": 1.0 - max(d8_similarities),
                }
                pool_rows.append({
                    "query_id": query_id,
                    "source_pool": pool,
                    "source_rank": int(item["source_rank"]),
                    "candidate_id": candidate_id,
                    "candidate_budget": budget,
                    "pool_construction_used_utility": False,
                    "pool_construction_used_mismatch_label": False,
                    "pool_construction_used_oracle_label": False,
                    "judgment_complete": judgment is not None,
                    "dense_rank": dense_rank,
                    "outside_dense100": dense_rank is None,
                    "cosine_similarity": score_context[query_id][candidate_id],
                    "within_query_similarity_percentile": (
                        percentile_context[query_id][candidate_id]
                    ),
                    "comment_word_count": len(_tokenize(corpus_text[candidate_id])),
                    **static,
                    **lexical,
                    "query_medication_lexicon_match": query_medication,
                    "candidate_medication_lexicon_match": candidate_medication,
                    "exact_medication_term_overlap": bool(
                        query_medication
                        and candidate_medication
                        and query_medication.lower() == candidate_medication.lower()
                    ),
                    "tool_term_overlap_status": "not_available_no_existing_lexicon",
                    "institution_term_overlap_status": (
                        "not_available_no_existing_lexicon"
                    ),
                    **route,
                    "native_graph": pool in {
                        POOL_GRAPH_PRACTICAL,
                        POOL_GRAPH_EXCLUSIVE,
                    },
                    "fallback_used": False,
                    "callback_used": False,
                    "padding_used": False,
                    "strict_provenance_source": (
                        "docs88_candidate_provenance"
                        if pool == POOL_GRAPH_EXCLUSIVE
                        else (
                            "raw_pre_fallback_route_plus_entry_trace"
                            if pool == POOL_GRAPH_PRACTICAL
                            else None
                        )
                    ),
                    "D8_set_internal_mean_similarity": d8_internal_mean,
                    "D8_set_internal_max_similarity": d8_internal_max,
                    "D8_set_similarity_dispersion": float(
                        np.std([
                            _cosine(d8_vectors[left], d8_vectors[right])
                            for left in range(len(d8_vectors))
                            for right in range(left + 1, len(d8_vectors))
                        ])
                    ),
                    **(
                        _dimensions(judgment)
                        if judgment is not None
                        else {short: None for short in (*DIMENSIONS, "U")}
                    ),
                    "safety_gate_status": (
                        "gated"
                        if judgment is not None
                        and float(judgment["label_safety"]) <= 2
                        else ("not_gated" if judgment is not None else "unjudged")
                    ),
                })
    return (
        pool_ids,
        dense_mismatch_rows,
        pool_rows,
        score_context,
    )


def _static_feature_registry(
    *,
    qids: set[str],
    dense_mismatch_rows: list[dict],
    pool_rows: list[dict],
    d8_ids: dict[str, list[str]],
    dense_rank: dict[str, dict[str, int]],
    candidate_vectors: dict[str, np.ndarray],
    query_vectors: dict[str, np.ndarray],
    corpus_text: dict[str, str],
    query_text: dict[str, str],
    idf: dict[str, float],
) -> dict[tuple[str, str], dict[str, float]]:
    output = {
        (str(row["query_id"]), str(row["candidate_id"])): {
            name: float(row[name]) for name in STATIC_PREDICTOR_FEATURES
        }
        for row in pool_rows
    }
    mismatch_by_pair = {
        (str(row["query_id"]), str(row["candidate_id"])): row
        for row in dense_mismatch_rows
    }
    for query_id in sorted(qids):
        vectors = [
            candidate_vectors[candidate_id] for candidate_id in d8_ids[query_id]
        ]
        for candidate_id in d8_ids[query_id]:
            vector = candidate_vectors[candidate_id]
            similarities = [_cosine(vector, other) for other in vectors]
            lexical = _lexical_diagnostics(
                query_text[query_id],
                corpus_text[candidate_id],
                idf,
            )
            rank = dense_rank[query_id][candidate_id]
            output[(query_id, candidate_id)] = {
                "query_candidate_sbert_similarity": float(
                    mismatch_by_pair[(query_id, candidate_id)][
                        "cosine_similarity"
                    ]
                ),
                "candidate_dense_percentile": (101 - rank) / 100.0,
                "candidate_dense_rank_missing": 0.0,
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
    return output


class _AuxiliaryUtilityModel:
    def __init__(self, scaler: StandardScaler, model: HuberRegressor) -> None:
        self.scaler = scaler
        self.model = model

    def predict(self, rows: list[dict[str, float]]) -> np.ndarray:
        matrix = np.asarray([
            [float(row[name]) for name in STATIC_PREDICTOR_FEATURES]
            for row in rows
        ], dtype=np.float64)
        return np.clip(
            self.model.predict(self.scaler.transform(matrix)),
            1.0,
            7.0,
        )


def _fit_auxiliary_utility_model(
    *,
    qids: list[str],
    candidate_ids: dict[str, list[str]],
    static_features: dict[tuple[str, str], dict[str, float]],
    registry: dict[tuple[str, str], dict],
    config: dict,
) -> _AuxiliaryUtilityModel:
    pairs = [
        (query_id, candidate_id)
        for query_id in qids
        for candidate_id in candidate_ids[query_id]
    ]
    if not pairs:
        raise ValueError("auxiliary utility model has no training pairs")
    matrix = np.asarray([
        [
            float(static_features[pair][name])
            for name in STATIC_PREDICTOR_FEATURES
        ]
        for pair in pairs
    ], dtype=np.float64)
    target = np.asarray(
        [float(registry[pair]["utility"]) for pair in pairs],
        dtype=np.float64,
    )
    scaler = StandardScaler().fit(matrix)
    model = HuberRegressor(
        epsilon=float(config["huber_epsilon"]),
        alpha=float(config["l2_alpha"]),
        max_iter=int(config["max_iter"]),
        fit_intercept=True,
    )
    model.fit(scaler.transform(matrix), target)
    return _AuxiliaryUtilityModel(scaler, model)


def _fold_local_predicted_utility(
    *,
    full_train_qids: list[str],
    full_valid_qids: list[str],
    inner_splits: list[tuple[list[str], list[str]]],
    eligible_qids: set[str],
    candidate_ids: dict[str, list[str]],
    static_features: dict[tuple[str, str], dict[str, float]],
    registry: dict[tuple[str, str], dict],
    config: dict,
) -> tuple[dict[tuple[str, str], float], dict]:
    train_qids = [query_id for query_id in full_train_qids if query_id in eligible_qids]
    valid_qids = [query_id for query_id in full_valid_qids if query_id in eligible_qids]
    predicted: dict[tuple[str, str], float] = {}
    inner_audit = []
    for inner_index, (full_inner_train, full_inner_valid) in enumerate(inner_splits):
        inner_train = [
            query_id for query_id in full_inner_train if query_id in eligible_qids
        ]
        inner_valid = [
            query_id for query_id in full_inner_valid if query_id in eligible_qids
        ]
        if not inner_valid:
            inner_audit.append({
                "inner_fold": inner_index,
                "train_queries": len(inner_train),
                "validation_queries": 0,
            })
            continue
        if not inner_train:
            raise ValueError("coverage filtering emptied an inner training fold")
        model = _fit_auxiliary_utility_model(
            qids=inner_train,
            candidate_ids=candidate_ids,
            static_features=static_features,
            registry=registry,
            config=config,
        )
        pairs = [
            (query_id, candidate_id)
            for query_id in inner_valid
            for candidate_id in candidate_ids[query_id]
        ]
        values = model.predict([static_features[pair] for pair in pairs])
        predicted.update(zip(pairs, map(float, values), strict=True))
        inner_audit.append({
            "inner_fold": inner_index,
            "train_queries": len(inner_train),
            "validation_queries": len(inner_valid),
            "query_overlap": len(set(inner_train) & set(inner_valid)),
        })
    expected_train_pairs = {
        (query_id, candidate_id)
        for query_id in train_qids
        for candidate_id in candidate_ids[query_id]
    }
    if set(predicted) != expected_train_pairs:
        missing = expected_train_pairs - set(predicted)
        raise ValueError(
            f"inner OOF predicted utility coverage failed: {len(missing)} pairs"
        )
    outer_model = _fit_auxiliary_utility_model(
        qids=train_qids,
        candidate_ids=candidate_ids,
        static_features=static_features,
        registry=registry,
        config=config,
    )
    outer_pairs = [
        (query_id, candidate_id)
        for query_id in valid_qids
        for candidate_id in candidate_ids[query_id]
    ]
    outer_values = outer_model.predict(
        [static_features[pair] for pair in outer_pairs]
    )
    predicted.update(zip(outer_pairs, map(float, outer_values), strict=True))
    return predicted, {
        "train_queries": len(train_qids),
        "validation_queries": len(valid_qids),
        "inner_splits": inner_audit,
        "outer_train_validation_overlap": len(set(train_qids) & set(valid_qids)),
        "prediction_is_in_sample_for_query": False,
    }


def _action_dimension_decomposition(
    candidate: dict,
    replaced: dict,
) -> dict:
    deltas = {
        short: (
            float(candidate[f"label_{name}"])
            - float(replaced[f"label_{name}"])
        )
        for short, name in DIMENSIONS.items()
    }
    contributions = {
        short: UTILITY_WEIGHTS[short] * deltas[short] / 8.0
        for short in UTILITY_WEIGHTS
    }
    gate_residual = (
        _safety_gate_residual(candidate) - _safety_gate_residual(replaced)
    ) / 8.0
    contributions_exact = dict(contributions)
    contributions_exact["S"] += gate_residual
    raw_delta = (
        float(candidate["utility"]) - float(replaced["utility"])
    ) / 8.0
    if not math.isclose(
        sum(contributions_exact.values()),
        raw_delta,
        abs_tol=1e-10,
    ):
        raise AssertionError("dimension decomposition is not exact")
    return {
        **{f"delta_{short}": value for short, value in deltas.items()},
        **{
            f"contribution_{short}": value
            for short, value in contributions_exact.items()
        },
        "safety_gate_residual": gate_residual,
        "candidate_safety_gated": float(candidate["label_safety"]) <= 2,
        "replaced_safety_gated": float(replaced["label_safety"]) <= 2,
        "safety_gated_case": (
            float(candidate["label_safety"]) <= 2
            or float(replaced["label_safety"]) <= 2
        ),
        "raw_utility_at8_delta": raw_delta,
    }


def _build_fold_actions(
    *,
    repeat: int,
    fold: int,
    train_qids: list[str],
    valid_qids: list[str],
    pool: str,
    pool_ids: dict[str, list[str]],
    d8_ids: dict[str, list[str]],
    drift_items: dict[str, list[str]],
    predicted_utility: dict[tuple[str, str], float],
    static_features: dict[tuple[str, str], dict[str, float]],
    pool_rows_by_pair: dict[tuple[str, str, str], dict],
    candidate_vectors: dict[str, np.ndarray],
    registry: dict[tuple[str, str], dict],
) -> tuple[list[dict], dict[str, dict]]:
    roles = {
        **{query_id: "outer_train_inner_oof" for query_id in train_qids},
        **{query_id: "outer_validation_oof" for query_id in valid_qids},
    }
    rows: list[dict] = []
    contexts: dict[str, dict] = {}
    for query_id in train_qids + valid_qids:
        baseline = d8_ids[query_id]
        baseline_utility = statistics.fmean(
            float(registry[(query_id, candidate_id)]["utility"])
            for candidate_id in baseline
        )
        contexts[query_id] = {
            "current_full_utility": baseline_utility,
        }
        rows.append({
            "repeat": repeat,
            "fold": fold,
            "query_id": query_id,
            "role": roles[query_id],
            "pool": pool,
            "action_id": "NOOP",
            "action_type": "NOOP",
            "candidate_id": None,
            "replaced_candidate_id": None,
            "replacement_rank": None,
            "eligible_all8": True,
            "model_features": {
                name: 0.0 for name in ROUTE_AWARE_FEATURES
            },
            "raw_reward_for_training_and_evaluation_only": 0.0,
            "baseline_utility_for_reward_only": baseline_utility,
            "action_utility_for_reward_only": baseline_utility,
            "fallback_used": False,
            "callback_used": False,
            "padding_used": False,
            "route_membership": "NOOP",
            "source_attribution_traceable": True,
            "inference_used_gold_utility": False,
            "eligibility_used_mismatch_label": False,
        })
        d8_vectors = [
            candidate_vectors[candidate_id] for candidate_id in baseline
        ]
        d8_pairwise = [
            _cosine(d8_vectors[left], d8_vectors[right])
            for left in range(len(d8_vectors))
            for right in range(left + 1, len(d8_vectors))
        ]
        d8_dispersion = float(np.std(d8_pairwise))
        d8_redundancy = statistics.fmean(d8_pairwise)
        for candidate_id in pool_ids[query_id]:
            candidate_static = static_features[(query_id, candidate_id)]
            candidate_vector = candidate_vectors[candidate_id]
            source_row = pool_rows_by_pair[(query_id, pool, candidate_id)]
            bm25_rank = source_row.get("bm25_rank")
            graph_ranks = dict(source_row.get("graph_pre_fallback_rank") or {})
            graph_relative = dict(source_row.get("graph_score_relative") or {})
            best_graph_rank = min(graph_ranks.values()) if graph_ranks else None
            # The realised selector must identify both whether to intervene and
            # which D8 item to replace.  Gold-derived A/B mismatch labels are
            # diagnostic only and never restrict action eligibility.
            for replaced_id in baseline:
                replacement_rank = baseline.index(replaced_id) + 1
                replaced_static = static_features[(query_id, replaced_id)]
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
                    "comment_length_log": candidate_static[
                        "comment_length_log"
                    ],
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
                        candidate_vector,
                        candidate_vectors[replaced_id],
                    ),
                    "oof_predicted_utility_difference": (
                        predicted_utility[(query_id, candidate_id)]
                        - predicted_utility[(query_id, replaced_id)]
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
                    "d8_similarity_dispersion": d8_dispersion,
                    "d8_redundancy": d8_redundancy,
                    "bm25_rank_norm": (
                        1.0 - (int(bm25_rank) - 1) / 99.0
                        if bm25_rank is not None else 0.0
                    ),
                    "bm25_score_relative": float(
                        source_row.get("bm25_score_relative") or 0.0
                    ),
                    "bm25_route_missing": float(bm25_rank is None),
                    "graph_ppr_rank_norm": (
                        1.0 - (int(best_graph_rank) - 1) / 99.0
                        if best_graph_rank is not None else 0.0
                    ),
                    "graph_ppr_score_relative": (
                        max(map(float, graph_relative.values()))
                        if graph_relative else 0.0
                    ),
                    "graph_route_missing": float(best_graph_rank is None),
                }
                if tuple(features) != ROUTE_AWARE_FEATURES:
                    raise AssertionError("action feature order drifted")
                if any(
                    forbidden in feature.lower()
                    for feature in SOURCE_BLIND_FEATURES
                    for forbidden in FORBIDDEN_INFERENCE_FEATURES
                ):
                    raise AssertionError("forbidden source-blind inference feature")
                candidate_judgment = registry[(query_id, candidate_id)]
                replaced_judgment = registry[(query_id, replaced_id)]
                dimension = _action_dimension_decomposition(
                    candidate_judgment,
                    replaced_judgment,
                )
                reward = float(dimension["raw_utility_at8_delta"])
                rows.append({
                    "repeat": repeat,
                    "fold": fold,
                    "query_id": query_id,
                    "role": roles[query_id],
                    "pool": pool,
                    "action_id": (
                        f"replace:r{replacement_rank}:{candidate_id}"
                    ),
                    "action_type": "REPLACE",
                    "candidate_id": candidate_id,
                    "replaced_candidate_id": replaced_id,
                    "replacement_rank": replacement_rank,
                    "eligible_all8": True,
                    "model_features": features,
                    "raw_reward_for_training_and_evaluation_only": reward,
                    "baseline_utility_for_reward_only": baseline_utility,
                    "action_utility_for_reward_only": baseline_utility + reward,
                    "fallback_used": bool(source_row["fallback_used"]),
                    "callback_used": bool(source_row["callback_used"]),
                    "padding_used": bool(source_row["padding_used"]),
                    "route_membership": pool,
                    "source_attribution_traceable": True,
                    "inference_used_gold_utility": False,
                    "eligibility_used_mismatch_label": False,
                    "mismatch_type_replaced": (
                        registry[(query_id, replaced_id)]["_mismatch_type"]
                    ),
                    **dimension,
                })
    return rows, contexts


def _oracle_rows(
    *,
    qids: set[str],
    pool_ids: dict[str, dict[str, list[dict]]],
    d8_ids: dict[str, list[str]],
    drift_items: dict[str, list[str]],
    registry: dict[tuple[str, str], dict],
    pool_rows_by_pair: dict[tuple[str, str, str], dict],
) -> list[dict]:
    output = []
    for query_id in sorted(qids):
        for pool in POOL_ORDER:
            candidates = [
                str(row["comment_id"]) for row in pool_ids[query_id][pool]
            ]
            judged = [
                candidate_id
                for candidate_id in candidates
                if (query_id, candidate_id) in registry
            ]
            actions = []
            for candidate_id in judged:
                for replaced_id in drift_items[query_id]:
                    decomposition = _action_dimension_decomposition(
                        registry[(query_id, candidate_id)],
                        registry[(query_id, replaced_id)],
                    )
                    actions.append({
                        "candidate_id": candidate_id,
                        "replaced_candidate_id": replaced_id,
                        "replacement_rank": (
                            d8_ids[query_id].index(replaced_id) + 1
                        ),
                        "candidate_utility_delta": (
                            float(registry[(query_id, candidate_id)]["utility"])
                            - float(registry[(query_id, replaced_id)]["utility"])
                        ),
                        **decomposition,
                    })
            positive = [
                row for row in actions
                if float(row["raw_utility_at8_delta"]) > 0
            ]
            best = max(
                positive,
                key=lambda row: (
                    float(row["raw_utility_at8_delta"]),
                    str(row["candidate_id"]),
                    -int(row["replacement_rank"]),
                ),
                default=None,
            )
            candidate_row = (
                pool_rows_by_pair[
                    (query_id, pool, str(best["candidate_id"]))
                ]
                if best is not None else None
            )
            output.append({
                "query_id": query_id,
                "source_pool": pool,
                "drift_query": bool(drift_items[query_id]),
                "drift_item_count": len(drift_items[query_id]),
                "candidate_budget": len(candidates),
                "judged_candidate_count": len(judged),
                "pool_judgment_complete": len(judged) == len(candidates),
                "observed_action_count": len(actions),
                "oracle_status": (
                    "coverage_complete"
                    if len(judged) == len(candidates)
                    else "judged_subset_lower_bound"
                ),
                "positive_rescue": best is not None,
                "material_rescue": (
                    best is not None
                    and float(best["candidate_utility_delta"]) >= 1.0
                ),
                "best_candidate_id": (
                    str(best["candidate_id"]) if best else None
                ),
                "best_replaced_candidate_id": (
                    str(best["replaced_candidate_id"]) if best else None
                ),
                "best_replacement_rank": (
                    int(best["replacement_rank"]) if best else None
                ),
                "best_candidate_outside_dense100": (
                    bool(candidate_row["outside_dense100"])
                    if candidate_row is not None else None
                ),
                "route_exclusive_positive_rescue": bool(
                    best is not None
                    and candidate_row is not None
                    and bool(candidate_row["outside_dense100"])
                ),
                "oracle_candidate_utility_delta": (
                    float(best["candidate_utility_delta"]) if best else 0.0
                ),
                "oracle_utility_at8_rescue": (
                    float(best["raw_utility_at8_delta"]) if best else 0.0
                ),
                **({
                    key: value
                    for key, value in best.items()
                    if key.startswith("delta_")
                    or key.startswith("contribution_")
                    or key in {
                        "safety_gate_residual",
                        "candidate_safety_gated",
                        "replaced_safety_gated",
                        "safety_gated_case",
                    }
                } if best else {
                    **{
                        f"delta_{short}": 0.0 for short in DIMENSIONS
                    },
                    **{
                        f"contribution_{short}": 0.0 for short in DIMENSIONS
                    },
                    "safety_gate_residual": 0.0,
                    "candidate_safety_gated": False,
                    "replaced_safety_gated": False,
                    "safety_gated_case": False,
                }),
            })
    return output


def _oracle_summary(
    rows: list[dict],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict, list[dict]]:
    summary = {}
    dimensions = []
    for pool_index, pool in enumerate(POOL_ORDER):
        pool_rows = [row for row in rows if row["source_pool"] == pool]
        complete = [
            row for row in pool_rows
            if row["pool_judgment_complete"] and row["drift_query"]
        ]
        lower_bound = [row for row in pool_rows if row["drift_query"]]
        values = [
            float(row["oracle_utility_at8_rescue"]) for row in complete
        ]
        observed_values = [
            float(row["oracle_utility_at8_rescue"]) for row in lower_bound
        ]
        item_counts = [
            int(row["drift_item_count"]) for row in complete
        ]
        summary[pool] = {
            "candidate_pool_queries": len(pool_rows),
            "candidate_pool_rows": sum(
                int(row["candidate_budget"]) for row in pool_rows
            ),
            "fully_judged_queries": sum(
                bool(row["pool_judgment_complete"]) for row in pool_rows
            ),
            "drift_queries_total": sum(
                bool(row["drift_query"]) for row in pool_rows
            ),
            "coverage_complete_drift_queries": len(complete),
            "mean_drift_items_per_covered_query": _mean(item_counts),
            "positive_rescue_queries": sum(
                bool(row["positive_rescue"]) for row in complete
            ),
            "positive_rescue_query_rate": (
                sum(bool(row["positive_rescue"]) for row in complete)
                / len(complete) if complete else None
            ),
            "material_rescue_queries": sum(
                bool(row["material_rescue"]) for row in complete
            ),
            "material_rescue_query_rate": (
                sum(bool(row["material_rescue"]) for row in complete)
                / len(complete) if complete else None
            ),
            "oracle_utility_at8": _bootstrap_summary(
                values,
                samples=bootstrap_samples,
                seed=bootstrap_seed + pool_index,
            ),
            "judged_subset_lower_bound_all_drift_queries": {
                "queries": len(lower_bound),
                "mean_oracle_utility_at8": _mean(observed_values),
                "warning": (
                    None
                    if all(row["pool_judgment_complete"] for row in lower_bound)
                    else "incomplete judgments; this is only a lower bound"
                ),
            },
            "route_exclusive_rescue_queries": sum(
                bool(row["route_exclusive_positive_rescue"])
                for row in complete
            ),
            "safety_gated_best_actions": sum(
                bool(row["safety_gated_case"]) for row in complete
            ),
        }
        positive = [row for row in complete if row["positive_rescue"]]
        dimension_row = {
            "source_pool": pool,
            "analysis_scope": "coverage_complete_positive_oracle_actions",
            "queries": len(positive),
        }
        for short in DIMENSIONS:
            dimension_row[f"mean_delta_{short}"] = _mean(
                float(row[f"delta_{short}"]) for row in positive
            )
            dimension_row[f"mean_contribution_{short}"] = _mean(
                float(row[f"contribution_{short}"]) for row in positive
            )
        dimension_row["mean_safety_gate_residual"] = _mean(
            float(row["safety_gate_residual"]) for row in positive
        )
        dimension_row["mean_oracle_utility_at8_rescue"] = _mean(
            float(row["oracle_utility_at8_rescue"]) for row in positive
        )
        dimensions.append(dimension_row)
    return summary, dimensions


def _paired_oracle_contrast(
    *,
    rows: list[dict],
    left: str,
    right: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    indexed = {
        (str(row["query_id"]), str(row["source_pool"])): row
        for row in rows
    }
    qids = sorted({
        str(row["query_id"])
        for row in rows
        if row["source_pool"] == left
        and row["drift_query"]
        and row["pool_judgment_complete"]
        and indexed.get((str(row["query_id"]), right), {}).get(
            "pool_judgment_complete"
        )
    })
    differences = [
        float(indexed[(query_id, left)]["oracle_utility_at8_rescue"])
        - float(indexed[(query_id, right)]["oracle_utility_at8_rescue"])
        for query_id in qids
    ]
    result = _bootstrap_summary(
        differences,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    result.update({
        "left": left,
        "right": right,
        "unit": "query",
        "matched_coverage_complete_drift_queries": len(qids),
        "win_tie_loss": _wtl(differences),
        "coverage_limited": len(qids) < 79,
    })
    return result


def _aggregate_oof_scope(rows: list[dict], qids: set[str]) -> dict:
    selected = [row for row in rows if str(row["query_id"]) in qids]
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_query[str(row["query_id"])].append(row)
    deltas = [
        statistics.fmean(float(row["raw_reward"]) for row in query_rows)
        for query_rows in by_query.values()
    ]
    actions = [
        statistics.fmean(float(row["acted"]) for row in query_rows)
        for query_rows in by_query.values()
    ]
    successes = [
        statistics.fmean(float(row["successful_action"]) for row in query_rows)
        for query_rows in by_query.values()
    ]
    harms = [
        statistics.fmean(float(row["harmful_action"]) for row in query_rows)
        for query_rows in by_query.values()
    ]
    oracle = [
        statistics.fmean(
            float(row["action_space_oracle_headroom"]) for row in query_rows
        )
        for query_rows in by_query.values()
    ]
    action_mass = sum(actions)
    success_mass = sum(successes)
    harm_mass = sum(harms)
    opportunity_queries = sum(value > 0 for value in oracle)
    return {
        "queries": len(by_query),
        "mean_delta": _mean(deltas),
        "values": deltas,
        "action_rate": _mean(actions),
        "successful_action_rate": _mean(successes),
        "harmful_action_rate": _mean(harms),
        "entrant_precision": (
            success_mass / action_mass if action_mass else None
        ),
        "rescue_opportunity_queries": opportunity_queries,
        "rescue_opportunity_recall": (
            success_mass / opportunity_queries
            if opportunity_queries else None
        ),
        "mean_oracle_headroom": _mean(oracle),
        "oracle_conversion_ratio": (
            statistics.fmean(deltas) / statistics.fmean(oracle)
            if oracle and statistics.fmean(oracle) else None
        ),
        "successful_action_mass": success_mass,
        "harmful_action_mass": harm_mass,
    }


def _realised_summary(
    *,
    rows: list[dict],
    complete_qids: dict[str, set[str]],
    drift_types: dict[str, set[str]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    output = {}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source_pool"]), str(row["feature_mode"]))].append(row)
    for index, ((pool, mode), arm_rows) in enumerate(sorted(grouped.items())):
        scopes = {
            "overall": set(complete_qids[pool]),
            "drift_query_only": {
                query_id for query_id in complete_qids[pool]
                if drift_types[query_id] & {MISMATCH_A, MISMATCH_B}
            },
            "semantic_context_mismatch_queries": {
                query_id for query_id in complete_qids[pool]
                if MISMATCH_A in drift_types[query_id]
            },
            "support_quality_mismatch_queries": {
                query_id for query_id in complete_qids[pool]
                if MISMATCH_B in drift_types[query_id]
            },
            "safety_mismatch_queries": {
                query_id for query_id in complete_qids[pool]
                if MISMATCH_C in drift_types[query_id]
            },
        }
        scope_output = {}
        for scope_index, (scope_name, qids) in enumerate(scopes.items()):
            values = _aggregate_oof_scope(arm_rows, qids)
            bootstrap = _bootstrap_summary(
                values.pop("values"),
                samples=bootstrap_samples,
                seed=bootstrap_seed + index * 20 + scope_index,
            )
            scope_output[scope_name] = {
                **values,
                "query_bootstrap_95ci": bootstrap[
                    "query_bootstrap_95ci"
                ],
            }
        output[f"{pool}__{mode}"] = {
            "source_pool": pool,
            "feature_mode": mode,
            "coverage_complete_queries": len(complete_qids[pool]),
            "scopes": scope_output,
        }
    return output


def _run_realised_policies(
    *,
    split_rows: list[dict],
    pool_ids: dict[str, dict[str, list[dict]]],
    complete_qids: dict[str, set[str]],
    d8_ids: dict[str, list[str]],
    drift_items: dict[str, list[str]],
    static_features: dict[tuple[str, str], dict[str, float]],
    pool_rows_by_pair: dict[tuple[str, str, str], dict],
    candidate_vectors: dict[str, np.ndarray],
    registry: dict[tuple[str, str], dict],
    policy: dict,
) -> tuple[list[dict], list[dict]]:
    output: list[dict] = []
    fold_audit: list[dict] = []
    inner_folds_count = int(policy["inner_folds"])
    for pool in POOL_ORDER:
        eligible = set(complete_qids[pool])
        candidate_ids = {
            query_id: [
                *d8_ids[query_id],
                *[
                    str(row["comment_id"])
                    for row in pool_ids[query_id][pool]
                ],
            ]
            for query_id in eligible
        }
        if any(len(set(ids)) != 12 for ids in candidate_ids.values()):
            raise ValueError(f"{pool}: expected D8 plus four distinct candidates")
        validation_count = Counter()
        for fold_row in split_rows:
            repeat = int(fold_row["repeat"])
            fold = int(fold_row["fold"])
            seed = int(fold_row["seed"])
            full_train = list(map(str, fold_row["train_query_ids"]))
            full_valid = list(map(str, fold_row["validation_query_ids"]))
            train_qids = [query_id for query_id in full_train if query_id in eligible]
            valid_qids = [query_id for query_id in full_valid if query_id in eligible]
            for query_id in valid_qids:
                validation_count[query_id] += 1
            if not valid_qids:
                fold_audit.append({
                    "source_pool": pool,
                    "repeat": repeat,
                    "fold": fold,
                    "skipped": True,
                    "reason": "no coverage-complete validation query in fold",
                    "train_queries": len(train_qids),
                    "validation_queries": 0,
                })
                continue
            full_inner_splits = canonical.inner_folds(
                full_train,
                inner_folds_count,
                seed + 7000,
            )
            inner_splits = [
                (
                    [query_id for query_id in inner_train if query_id in eligible],
                    [query_id for query_id in inner_valid if query_id in eligible],
                )
                for inner_train, inner_valid in full_inner_splits
            ]
            predicted, prediction_audit = _fold_local_predicted_utility(
                full_train_qids=full_train,
                full_valid_qids=full_valid,
                inner_splits=full_inner_splits,
                eligible_qids=eligible,
                candidate_ids=candidate_ids,
                static_features=static_features,
                registry=registry,
                config=dict(policy["auxiliary_utility_model"]),
            )
            action_rows, contexts = _build_fold_actions(
                repeat=repeat,
                fold=fold,
                train_qids=train_qids,
                valid_qids=valid_qids,
                pool=pool,
                pool_ids={
                    query_id: [
                        str(row["comment_id"])
                        for row in pool_ids[query_id][pool]
                    ]
                    for query_id in eligible
                },
                d8_ids=d8_ids,
                drift_items=drift_items,
                predicted_utility=predicted,
                static_features=static_features,
                pool_rows_by_pair=pool_rows_by_pair,
                candidate_vectors=candidate_vectors,
                registry=registry,
            )
            fold_record = {
                "source_pool": pool,
                "repeat": repeat,
                "fold": fold,
                "skipped": False,
                "train_queries": len(train_qids),
                "validation_queries": len(valid_qids),
                "train_validation_overlap": len(
                    set(train_qids) & set(valid_qids)
                ),
                "full_report89_train_queries": len(full_train),
                "full_report89_validation_queries": len(full_valid),
                "coverage_filter_only": len(eligible) < 100,
                "auxiliary_prediction": prediction_audit,
                "feature_modes": {},
            }
            for mode, features in (
                ("source_blind", SOURCE_BLIND_FEATURES),
                ("route_aware", ROUTE_AWARE_FEATURES),
            ):
                predictions, tuning = _evaluate_fold_variant(
                    action_rows,
                    pool=pool,
                    action_space="all8",
                    family="direct_delta",
                    train_qids=train_qids,
                    valid_qids=valid_qids,
                    inner_splits=inner_splits,
                    model_config=dict(policy["direct_delta"]),
                    kappa_options=list(map(float, policy["kappa_options"])),
                    threshold_quantiles=list(
                        map(float, policy["threshold_quantiles"])
                    ),
                    seed=seed + (0 if mode == "source_blind" else 500000),
                    baseline_contexts=contexts,
                    feature_names=tuple(features),
                )
                for row in predictions:
                    row["source_pool"] = pool
                    row["feature_mode"] = mode
                    row["arm"] = (
                        f"{pool}__{mode}__direct_delta__"
                        f"{row['threshold_mode']}"
                    )
                    row["candidate_space_judgment_complete"] = True
                    row["coverage_complete_query_count"] = len(eligible)
                output.extend(predictions)
                fold_record["feature_modes"][mode] = tuning
            fold_audit.append(fold_record)
        if set(validation_count) != eligible or any(
            count != 5 for count in validation_count.values()
        ):
            raise ValueError(
                f"{pool}: filtered report89 validation membership changed"
            )
    return output, fold_audit


def _route_aware_contrasts(
    rows: list[dict],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    by_key: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_key[
            (
                str(row["source_pool"]),
                str(row["feature_mode"]),
                str(row["query_id"]),
            )
        ].append(float(row["raw_reward"]))
    output = {}
    for pool_index, pool in enumerate(POOL_ORDER):
        query_ids = sorted({
            query_id
            for source_pool, mode, query_id in by_key
            if source_pool == pool and mode == "source_blind"
            and (pool, "route_aware", query_id) in by_key
        })
        differences = [
            statistics.fmean(by_key[(pool, "route_aware", query_id)])
            - statistics.fmean(by_key[(pool, "source_blind", query_id)])
            for query_id in query_ids
        ]
        output[pool] = {
            **_bootstrap_summary(
                differences,
                samples=bootstrap_samples,
                seed=bootstrap_seed + pool_index,
            ),
            "contrast": "route_aware_minus_source_blind",
            "win_tie_loss": _wtl(differences),
        }
    return output


def _render_similarity_utility_scatter(
    path: Path,
    *,
    dense_rows: list[dict],
    pool_rows: list[dict],
    oracle_rows: list[dict],
) -> None:
    source_map = {
        POOL_DEEP: "Deep",
        POOL_BM25_PRACTICAL: "BM25",
        POOL_BM25_EXCLUSIVE: "BM25",
        POOL_GRAPH_PRACTICAL: "Graph",
        POOL_GRAPH_EXCLUSIVE: "Graph",
    }
    records = [
        {
            "source": "D8",
            "x": float(row["within_query_similarity_percentile"]),
            "y": float(row["U"]),
            "mismatch": row["mismatch_type"] in {MISMATCH_A, MISMATCH_B},
            "pair": (str(row["query_id"]), str(row["candidate_id"])),
        }
        for row in dense_rows
    ]
    seen = set()
    for row in pool_rows:
        if not row["judgment_complete"]:
            continue
        source = source_map[str(row["source_pool"])]
        key = (
            str(row["query_id"]),
            str(row["candidate_id"]),
            source,
        )
        if key in seen:
            continue
        seen.add(key)
        records.append({
            "source": source,
            "x": float(row["within_query_similarity_percentile"]),
            "y": float(row["U"]),
            "mismatch": False,
            "pair": (str(row["query_id"]), str(row["candidate_id"])),
        })
    rescue_pairs = {
        (str(row["query_id"]), str(row["best_candidate_id"]))
        for row in oracle_rows
        if row["positive_rescue"] and row["best_candidate_id"]
    }
    colors = {
        "D8": (51, 65, 85),
        "Deep": (37, 99, 235),
        "BM25": (217, 119, 6),
        "Graph": (124, 58, 237),
    }
    width, height = 1500, 980
    left, right, top, bottom = 120, 70, 110, 120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    def xy(x_value: float, y_value: float) -> tuple[int, int]:
        x = left + x_value * (width - left - right)
        y = top + (7.2 - y_value) / (7.2 - 0.8) * (
            height - top - bottom
        )
        return int(x), int(y)

    for tick in range(6):
        x = left + tick / 5 * (width - left - right)
        draw.line((x, top, x, height - bottom), fill=(148, 163, 184, 45))
        draw.text((x - 12, height - bottom + 12), f"{tick/5:.1f}",
                  fill=(51, 65, 85), font=font)
    for utility in range(1, 8):
        _, y = xy(0.0, float(utility))
        draw.line((left, y, width - right, y), fill=(148, 163, 184, 45))
        draw.text((left - 28, y - 6), str(utility),
                  fill=(51, 65, 85), font=font)
    _, y4 = xy(0.0, 4.0)
    for x in range(left, width - right, 16):
        draw.line((x, y4, min(x + 8, width - right), y4),
                  fill=(100, 116, 139, 180), width=2)

    for row in records:
        x, y = xy(row["x"], row["y"])
        color = (*colors[row["source"]], 80)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        if row["mismatch"]:
            draw.ellipse(
                (x - 7, y - 7, x + 7, y + 7),
                outline=(220, 38, 38, 230),
                width=2,
            )
        if row["pair"] in rescue_pairs and row["source"] != "D8":
            draw.line((x - 7, y, x + 7, y), fill=(5, 150, 105, 240), width=3)
            draw.line((x, y - 7, x, y + 7), fill=(5, 150, 105, 240), width=3)
    draw.rectangle((left, top, width - right, height - bottom),
                   outline=(51, 65, 85, 220), width=2)
    draw.text(
        (left, 34),
        "Semantic similarity and evidence utility",
        fill=(15, 23, 42),
        font=font,
    )
    draw.text(
        (left, 58),
        "Judged development candidates; red ring = Dense A/B mismatch; "
        "green cross = observed best positive rescue",
        fill=(71, 85, 105),
        font=font,
    )
    draw.text(
        (width // 2 - 160, height - 42),
        "Within-query SBERT similarity percentile (full corpus)",
        fill=(15, 23, 42),
        font=font,
    )
    draw.text((20, top - 25), "utility-v2", fill=(15, 23, 42), font=font)
    legend_x, legend_y = width - 420, top + 20
    for index, source in enumerate(("D8", "Deep", "BM25", "Graph")):
        y = legend_y + index * 24
        draw.ellipse(
            (legend_x, y, legend_x + 10, y + 10),
            fill=(*colors[source], 210),
        )
        draw.text((legend_x + 18, y - 2), source,
                  fill=(15, 23, 42), font=font)
    image.save(path, "PDF", resolution=150.0)


def _pool_inventory(
    *,
    qids: set[str],
    pool_ids: dict[str, dict[str, list[dict]]],
    pool_rows: list[dict],
) -> dict:
    by_pool: dict[str, list[dict]] = defaultdict(list)
    for row in pool_rows:
        by_pool[str(row["source_pool"])].append(row)
    output = {}
    for pool in POOL_ORDER:
        rows = by_pool[pool]
        per_query = Counter(str(row["query_id"]) for row in rows)
        dense_ranks = [
            int(row["dense_rank"])
            for row in rows if row["dense_rank"] is not None
        ]
        practical_or_exclusive_overlap = None
        if pool in {POOL_BM25_PRACTICAL, POOL_BM25_EXCLUSIVE}:
            counterpart = (
                POOL_BM25_EXCLUSIVE
                if pool == POOL_BM25_PRACTICAL else POOL_BM25_PRACTICAL
            )
            overlap = sum(
                len(
                    {
                        str(row["comment_id"])
                        for row in pool_ids[query_id][pool]
                    }
                    & {
                        str(row["comment_id"])
                        for row in pool_ids[query_id][counterpart]
                    }
                )
                for query_id in qids
            )
            practical_or_exclusive_overlap = overlap / len(rows)
        if pool in {POOL_GRAPH_PRACTICAL, POOL_GRAPH_EXCLUSIVE}:
            counterpart = (
                POOL_GRAPH_EXCLUSIVE
                if pool == POOL_GRAPH_PRACTICAL else POOL_GRAPH_PRACTICAL
            )
            overlap = sum(
                len(
                    {
                        str(row["comment_id"])
                        for row in pool_ids[query_id][pool]
                    }
                    & {
                        str(row["comment_id"])
                        for row in pool_ids[query_id][counterpart]
                    }
                )
                for query_id in qids
            )
            practical_or_exclusive_overlap = overlap / len(rows)
        output[pool] = {
            "candidate_rows": len(rows),
            "queries": len(per_query),
            "per_query_candidate_count_distribution": dict(
                sorted(Counter(per_query.values()).items())
            ),
            "within_pool_duplicate_pairs": (
                len(rows)
                - len({
                    (str(row["query_id"]), str(row["candidate_id"]))
                    for row in rows
                })
            ),
            "D8_overlap_rows": 0,
            "outside_D100_rows": sum(
                bool(row["outside_dense100"]) for row in rows
            ),
            "dense_rank_observed_rows": len(dense_ranks),
            "dense_rank_min": min(dense_ranks) if dense_ranks else None,
            "dense_rank_median": (
                float(np.median(dense_ranks)) if dense_ranks else None
            ),
            "dense_rank_max": max(dense_ranks) if dense_ranks else None,
            "judged_candidate_rows": sum(
                bool(row["judgment_complete"]) for row in rows
            ),
            "fully_judged_queries": sum(
                all(
                    row["judgment_complete"]
                    for row in rows
                    if str(row["query_id"]) == query_id
                )
                for query_id in qids
            ),
            "practical_exclusive_pair_overlap_rate": (
                practical_or_exclusive_overlap
            ),
        }
    return output


def _residual_rows(
    *,
    pools: set[str],
    pool_rows: list[dict],
    query_text: dict[str, str],
    corpus_text: dict[str, str],
) -> list[dict]:
    grouped: dict[tuple[str, str], dict] = {}
    for row in pool_rows:
        if (
            str(row["source_pool"]) not in pools
            or bool(row["judgment_complete"])
        ):
            continue
        key = (str(row["query_id"]), str(row["candidate_id"]))
        record = grouped.setdefault(key, {
            "query_id": key[0],
            "comment_id": key[1],
            "query_text": query_text[key[0]],
            "comment_text": corpus_text[key[1]],
            "requested_rubric": "utility-v2-existing-protocol",
            "source_pools": [],
            "source_ranks": {},
            "external_call_authorised": False,
            "test_split": False,
        })
        record["source_pools"].append(str(row["source_pool"]))
        record["source_ranks"][str(row["source_pool"])] = int(
            row["source_rank"]
        )
    for row in grouped.values():
        row["source_pools"] = sorted(set(row["source_pools"]))
    return [grouped[key] for key in sorted(grouped)]


def _per_query_rows(
    *,
    qids: set[str],
    oracle_rows: list[dict],
    oof_rows: list[dict],
    drift_types: dict[str, set[str]],
    drift_items: dict[str, list[str]],
) -> list[dict]:
    oracle = {
        (str(row["query_id"]), str(row["source_pool"])): row
        for row in oracle_rows
    }
    grouped_oof: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in oof_rows:
        if (
            row["feature_mode"] == "source_blind"
            and row["threshold_mode"] == "nested"
        ):
            grouped_oof[
                (str(row["query_id"]), str(row["source_pool"]))
            ].append(row)
    output = []
    for query_id in sorted(qids):
        for pool in POOL_ORDER:
            oracle_row = oracle[(query_id, pool)]
            realised = grouped_oof.get((query_id, pool), [])
            output.append({
                "query_id": query_id,
                "source_pool": pool,
                "mismatch_A_present": MISMATCH_A in drift_types[query_id],
                "mismatch_B_present": MISMATCH_B in drift_types[query_id],
                "mismatch_C_present": MISMATCH_C in drift_types[query_id],
                "drift_query": bool(
                    drift_types[query_id] & {MISMATCH_A, MISMATCH_B}
                ),
                "drift_item_count": len(drift_items[query_id]),
                "candidate_budget": int(oracle_row["candidate_budget"]),
                "judged_candidate_count": int(
                    oracle_row["judged_candidate_count"]
                ),
                "pool_judgment_complete": bool(
                    oracle_row["pool_judgment_complete"]
                ),
                "oracle_status": oracle_row["oracle_status"],
                "oracle_utility_at8_rescue": float(
                    oracle_row["oracle_utility_at8_rescue"]
                ),
                "positive_rescue": bool(oracle_row["positive_rescue"]),
                "material_rescue": bool(oracle_row["material_rescue"]),
                "realised_nested_oof_available": bool(realised),
                "realised_source_blind_mean_delta": (
                    statistics.fmean(
                        float(row["raw_reward"]) for row in realised
                    )
                    if realised else None
                ),
                "realised_source_blind_action_rate": (
                    statistics.fmean(
                        float(row["acted"]) for row in realised
                    )
                    if realised else None
                ),
            })
    return output


def _resolve_inputs(
    config_key: str,
) -> tuple[Path, dict, dict[str, Path]]:
    root = Path(__file__).resolve().parents[1]
    all_config = project_config.load()
    policy = dict(all_config[config_key])
    source = dict(all_config[str(policy["source_experiment_key"])])
    source_policy = dict(all_config[str(policy["source_policy_key"])])
    paths = {
        "output_dir": _resolve(root, policy["output_dir"]),
        "candidate_pool": _resolve(root, source["candidate_pool"]),
        "utility_registry": _resolve(root, source["utility_registry"]),
        "dense_run": _resolve(root, source["dense_run"]),
        "source_output_dir": _resolve(root, source["output_dir"]),
        "split_manifest": (
            _resolve(root, source_policy["output_dir"]) / "split_manifest.json"
        ),
        "provenance_audit_dir": _resolve(
            root, policy["provenance_audit_dir"]
        ),
        "bm25_run": _resolve(root, policy["bm25_run"]),
        "bm25_sidecar_manifest": _resolve(
            root, policy["bm25_sidecar_manifest"]
        ),
        "query_admin": _resolve(root, policy["query_admin"]),
        "corpus": _resolve(root, source_policy["corpus"]),
        "corpus_embeddings": _resolve(
            root, source_policy["corpus_embeddings"]
        ),
        "queries": _resolve(root, source_policy["queries"]),
        "query_embeddings": _resolve(
            root, source_policy["query_embeddings"]
        ),
    }
    for route, value in dict(policy["graph_runs"]).items():
        paths[f"graph_run::{route}"] = _resolve(root, value)
    for route, value in dict(policy["graph_entry_traces"]).items():
        paths[f"graph_trace::{route}"] = _resolve(root, value)
    for path in paths.values():
        _reject_test(path)
    return root, policy, paths


def _verdicts(
    *,
    oracle_contrasts: dict,
    oracle_summary: dict,
    realised: dict,
    thresholds: dict,
) -> dict:
    bm25_vs_deep = oracle_contrasts["BM25-practical4_minus_Deep4"]
    graph_vs_deep = oracle_contrasts["Graph-practical4_minus_Deep4"]
    graph_vs_bm25 = oracle_contrasts[
        "Graph-exclusive4_minus_BM25-exclusive4"
    ]
    rule_a = bool(
        bm25_vs_deep["query_bootstrap_95ci"][0] is not None
        and bm25_vs_deep["query_bootstrap_95ci"][0] > 0
    )
    rule_b = bool(
        graph_vs_deep["query_bootstrap_95ci"][0] is not None
        and graph_vs_deep["query_bootstrap_95ci"][0] > 0
        and graph_vs_bm25["query_bootstrap_95ci"][0] is not None
        and graph_vs_bm25["query_bootstrap_95ci"][0] > 0
    )
    rule_c = bool(
        not rule_a
        and not rule_b
        and (
            bm25_vs_deep["mean"] is None
            or bm25_vs_deep["mean"] <= 0
            or bm25_vs_deep["query_bootstrap_95ci"][0] <= 0
        )
        and (
            graph_vs_deep["mean"] is None
            or graph_vs_deep["mean"] <= 0
            or graph_vs_deep["query_bootstrap_95ci"][0] <= 0
        )
    )
    conversions = []
    for pool in POOL_ORDER:
        key = f"{pool}__source_blind"
        scope = realised[key]["scopes"]["drift_query_only"]
        if scope["queries"]:
            conversions.append(scope)
    rule_d = any(
        scope["mean_oracle_headroom"] is not None
        and scope["mean_oracle_headroom"] > 0
        and scope["query_bootstrap_95ci"][0] <= 0
        and scope["query_bootstrap_95ci"][1] >= 0
        and (
            scope["oracle_conversion_ratio"] is None
            or abs(scope["oracle_conversion_ratio"])
            < float(thresholds["low_oracle_conversion_ratio"])
        )
        for scope in conversions
    )
    oracle_means = [
        oracle_summary[pool]["oracle_utility_at8"]["mean"]
        for pool in POOL_ORDER
        if oracle_summary[pool]["oracle_utility_at8"]["mean"] is not None
    ]
    rule_e = bool(
        oracle_means
        and all(
            value < float(thresholds["weak_oracle_utility_at8"])
            for value in oracle_means
        )
    )
    triggered = []
    for enabled, verdict in (
        (rule_a, "BM25_PARTIALLY_RESCUES_DENSE_LEXICAL_DRIFT"),
        (rule_b, "GRAPH_PARTIALLY_RESCUES_RELATIONAL_UNDERREACH"),
        (rule_c, "DEEPER_DENSE_IS_SUFFICIENT_FOR_OBSERVED_DRIFT"),
        (
            rule_d,
            "ALTERNATIVE_ROUTES_ADD_RESCUE_CANDIDATES_BUT_SELECTION_REMAINS_THE_BOTTLENECK",
        ),
        (
            rule_e,
            "OBSERVED_DENSE_MISMATCH_IS_NOT_RESOLVED_BY_TESTED_ALTERNATIVE_ROUTES",
        ),
    ):
        if enabled:
            triggered.append(verdict)
    return {
        "A": {"triggered": rule_a},
        "B": {"triggered": rule_b},
        "C": {"triggered": rule_c},
        "D": {"triggered": rule_d},
        "E": {"triggered": rule_e},
        "triggered_verdicts": triggered,
        "thresholds": thresholds,
    }


def run(
    config_key: str = "dense_semantic_drift_rescue_audit",
) -> dict:
    root, policy, paths = _resolve_inputs(config_key)
    output_dir = paths["output_dir"]
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite versioned output: {output_dir}"
        )
    required = [path for name, path in paths.items() if name != "output_dir"]
    required.extend([
        paths["provenance_audit_dir"] / "strict_native_graph_pool.jsonl",
        paths["provenance_audit_dir"] / "strict_native_graph_audit_report.json",
        paths["source_output_dir"] / "mixed_candidate_feature_registry.jsonl",
    ])
    for path in required:
        _reject_test(path)
        if not path.exists():
            raise FileNotFoundError(path)

    feature_rows = _read_jsonl(
        paths["source_output_dir"] / "mixed_candidate_feature_registry.jsonl"
    )
    qids = {str(row["query_id"]) for row in feature_rows}
    if len(qids) != 100 or len(feature_rows) != 1600:
        raise ValueError("audit expects the frozen 100×16 candidate registry")
    _, registry = complete_utility_v2_rows(
        _read_jsonl(paths["utility_registry"])
    )
    dense = _normalise_structured_run(paths["dense_run"], qids)
    bm25 = _normalise_flat_run(paths["bm25_run"], qids)
    graph_routes = {
        route: _normalise_structured_run(
            paths[f"graph_run::{route}"], qids
        )
        for route in policy["graph_runs"]
    }
    traces = {}
    for route in policy["graph_entry_traces"]:
        trace_rows = {
            str(row["query_id"]): row
            for row in _read_jsonl(paths[f"graph_trace::{route}"])
            if str(row["query_id"]) in qids
        }
        if set(trace_rows) != qids or any(
            int(row.get("selected_fact_count", 0)) <= 0
            for row in trace_rows.values()
        ):
            raise ValueError(f"{route}: pre-fallback graph entry trace failed")
        traces[route] = trace_rows
    strict_graph_rows = _read_jsonl(
        paths["provenance_audit_dir"] / "strict_native_graph_pool.jsonl"
    )
    if len(strict_graph_rows) != 400:
        raise ValueError("docs88 strict Graph4 must contain 400 rows")

    (
        candidate_vectors,
        query_vectors,
        corpus_text,
        query_text,
        corpus_matrix,
        corpus_ids,
    ) = _load_embeddings(
        corpus_path=paths["corpus"],
        corpus_embeddings_path=paths["corpus_embeddings"],
        queries_path=paths["queries"],
        query_embeddings_path=paths["query_embeddings"],
        qids=qids,
    )
    idf, idf_manifest = _build_idf(corpus_text.values())
    budget = int(policy["candidate_budget_per_source"])
    (
        pool_ids,
        dense_mismatch_rows,
        pool_rows,
        score_context,
    ) = _build_candidate_pools(
        qids=qids,
        feature_rows=feature_rows,
        dense=dense,
        bm25=bm25,
        graph_routes=graph_routes,
        strict_graph_rows=strict_graph_rows,
        registry=registry,
        candidate_vectors=candidate_vectors,
        query_vectors=query_vectors,
        corpus_matrix=corpus_matrix,
        corpus_text=corpus_text,
        query_text=query_text,
        idf=idf,
        budget=budget,
    )

    mismatch_cfg = dict(policy["mismatch"])
    drift_types: dict[str, set[str]] = defaultdict(set)
    drift_items: dict[str, list[str]] = defaultdict(list)
    d8_ids: dict[str, list[str]] = defaultdict(list)
    for row in dense_mismatch_rows:
        query_id = str(row["query_id"])
        candidate_id = str(row["candidate_id"])
        mismatch = _mismatch_type(registry[(query_id, candidate_id)], mismatch_cfg)
        row["mismatch_type"] = mismatch
        registry[(query_id, candidate_id)]["_mismatch_type"] = mismatch
        d8_ids[query_id].append(candidate_id)
        if mismatch:
            drift_types[query_id].add(mismatch)
        if mismatch in {MISMATCH_A, MISMATCH_B}:
            drift_items[query_id].append(candidate_id)
    for query_id in qids:
        d8_ids[query_id] = [
            str(row["comment_id"])
            for row in sorted(
                [
                    row for row in feature_rows
                    if str(row["query_id"]) == query_id
                    and str(row["candidate_source"]) == "dense_top8"
                ],
                key=lambda row: int(row["source_rank"]),
            )
        ]
        drift_items[query_id].sort(
            key=lambda candidate_id: d8_ids[query_id].index(candidate_id)
        )
        dense_reported = [
            str(row["comment_id"]) for row in dense[query_id][:8]
        ]
        if dense_reported != d8_ids[query_id]:
            raise ValueError(f"{query_id}: frozen D8 changed")

    dense_rank = {
        query_id: {
            str(row["comment_id"]): int(row["rank"])
            for row in dense[query_id]
        }
        for query_id in qids
    }
    static_features = _static_feature_registry(
        qids=qids,
        dense_mismatch_rows=dense_mismatch_rows,
        pool_rows=pool_rows,
        d8_ids=d8_ids,
        dense_rank=dense_rank,
        candidate_vectors=candidate_vectors,
        query_vectors=query_vectors,
        corpus_text=corpus_text,
        query_text=query_text,
        idf=idf,
    )
    pool_rows_by_pair = {
        (
            str(row["query_id"]),
            str(row["source_pool"]),
            str(row["candidate_id"]),
        ): row
        for row in pool_rows
    }
    pool_inventory = _pool_inventory(
        qids=qids,
        pool_ids=pool_ids,
        pool_rows=pool_rows,
    )
    complete_qids = {
        pool: {
            query_id for query_id in qids
            if all(
                (query_id, str(row["comment_id"])) in registry
                for row in pool_ids[query_id][pool]
            )
        }
        for pool in POOL_ORDER
    }
    oracle_rows = _oracle_rows(
        qids=qids,
        pool_ids=pool_ids,
        d8_ids=d8_ids,
        drift_items=drift_items,
        registry=registry,
        pool_rows_by_pair=pool_rows_by_pair,
    )
    oracle_summary, dimension_rows = _oracle_summary(
        oracle_rows,
        bootstrap_samples=int(policy["bootstrap_samples"]),
        bootstrap_seed=int(policy["bootstrap_seed"]),
    )
    oracle_contrasts = {
        "BM25-practical4_minus_Deep4": _paired_oracle_contrast(
            rows=oracle_rows,
            left=POOL_BM25_PRACTICAL,
            right=POOL_DEEP,
            bootstrap_samples=int(policy["bootstrap_samples"]),
            bootstrap_seed=int(policy["bootstrap_seed"]) + 100,
        ),
        "Graph-practical4_minus_Deep4": _paired_oracle_contrast(
            rows=oracle_rows,
            left=POOL_GRAPH_PRACTICAL,
            right=POOL_DEEP,
            bootstrap_samples=int(policy["bootstrap_samples"]),
            bootstrap_seed=int(policy["bootstrap_seed"]) + 200,
        ),
        "Graph-exclusive4_minus_BM25-exclusive4": _paired_oracle_contrast(
            rows=oracle_rows,
            left=POOL_GRAPH_EXCLUSIVE,
            right=POOL_BM25_EXCLUSIVE,
            bootstrap_samples=int(policy["bootstrap_samples"]),
            bootstrap_seed=int(policy["bootstrap_seed"]) + 300,
        ),
    }

    split_manifest = json.loads(
        paths["split_manifest"].read_text(encoding="utf-8")
    )
    split_rows = list(split_manifest["rows"])
    if len(split_rows) != 25:
        raise ValueError("report89 split manifest must contain 25 outer folds")
    oof_rows, fold_audit = _run_realised_policies(
        split_rows=split_rows,
        pool_ids=pool_ids,
        complete_qids=complete_qids,
        d8_ids=d8_ids,
        drift_items=drift_items,
        static_features=static_features,
        pool_rows_by_pair=pool_rows_by_pair,
        candidate_vectors=candidate_vectors,
        registry=registry,
        policy=policy,
    )
    nested_oof = [
        row for row in oof_rows if row["threshold_mode"] == "nested"
    ]
    realised_summary = _realised_summary(
        rows=nested_oof,
        complete_qids=complete_qids,
        drift_types=drift_types,
        bootstrap_samples=int(policy["bootstrap_samples"]),
        bootstrap_seed=int(policy["bootstrap_seed"]) + 1000,
    )
    route_contrasts = _route_aware_contrasts(
        nested_oof,
        bootstrap_samples=int(policy["bootstrap_samples"]),
        bootstrap_seed=int(policy["bootstrap_seed"]) + 2000,
    )

    bm25_residual = _residual_rows(
        pools={POOL_BM25_PRACTICAL, POOL_BM25_EXCLUSIVE},
        pool_rows=pool_rows,
        query_text=query_text,
        corpus_text=corpus_text,
    )
    graph_residual = _residual_rows(
        pools={POOL_GRAPH_PRACTICAL},
        pool_rows=pool_rows,
        query_text=query_text,
        corpus_text=corpus_text,
    )
    per_query = _per_query_rows(
        qids=qids,
        oracle_rows=oracle_rows,
        oof_rows=nested_oof,
        drift_types=drift_types,
        drift_items=drift_items,
    )

    bm25_sidecar = json.loads(
        paths["bm25_sidecar_manifest"].read_text(encoding="utf-8")
    )
    if bm25_sidecar["integrity"]["per_query_length_min"] != 100:
        raise ValueError("BM25 sidecar completeness changed")
    if _sha256(paths["bm25_run"]) != bm25_sidecar["source_run"]["sha256"]:
        raise ValueError("BM25 run SHA does not match frozen sidecar")
    mismatch_counts = Counter(
        row["mismatch_type"] or "none" for row in dense_mismatch_rows
    )
    mismatch_query_counts = {
        mismatch: sum(mismatch in drift_types[query_id] for query_id in qids)
        for mismatch in (MISMATCH_A, MISMATCH_B, MISMATCH_C)
    }
    artefact_inventory = {
        "schema": "dense-semantic-drift-rescue-artefact-inventory-v1",
        "status": "COMPLETE_LOCAL_NO_EXTERNAL_CALLS",
        "queries": len(qids),
        "bm25": {
            "run": str(paths["bm25_run"].relative_to(root)),
            "run_sha256": _sha256(paths["bm25_run"]),
            "sidecar": str(
                paths["bm25_sidecar_manifest"].relative_to(root)
            ),
            "rows": sum(len(rows) for rows in bm25.values()),
            "queries": len(bm25),
            "top100_complete_queries": sum(
                len(rows) == 100 for rows in bm25.values()
            ),
            "query_candidate_id_mapping": "exact query_id/comment_id",
            "preprocessing": {
                "backend": "bm25s",
                "method": "robertson",
                "k1": 1.5,
                "b": 0.75,
                "token_pattern": r"(?u)\b\w\w+\b",
                "stopwords": "english",
                "stemming": False,
                "canonical_implementation": (
                    "retrieval/section17_pipeline.py::BM25SCommentIndex"
                ),
            },
            "fallback_used": False,
            "padding_used": False,
            "provenance_ambiguity": False,
            "judgment_coverage": {
                pool: {
                    "candidate_rows": pool_inventory[pool]["candidate_rows"],
                    "judged_candidate_rows": pool_inventory[pool][
                        "judged_candidate_rows"
                    ],
                    "fully_judged_queries": pool_inventory[pool][
                        "fully_judged_queries"
                    ],
                }
                for pool in (POOL_BM25_PRACTICAL, POOL_BM25_EXCLUSIVE)
            },
            "residual_unique_pairs": len(bm25_residual),
        },
        "graph_practical": {
            "routes": sorted(graph_routes),
            "entry_trace_queries_with_positive_fact_seed_count": {
                route: sum(
                    int(row["selected_fact_count"]) > 0
                    for row in traces[route].values()
                )
                for route in traces
            },
            "fallback_used": False,
            "callback_used": False,
            "padding_used": False,
            "residual_unique_pairs": len(graph_residual),
        },
        "pool_inventory": pool_inventory,
        "idf_diagnostics": idf_manifest,
        "existing_term_lexicons": {
            "medication": "evaluation/safety_filter.py::DEFAULT_DRUG_PATTERNS",
            "tool": None,
            "institution": None,
        },
        "test_read": False,
        "external_model_calls": 0,
    }
    verdicts = _verdicts(
        oracle_contrasts=oracle_contrasts,
        oracle_summary=oracle_summary,
        realised=realised_summary,
        thresholds=dict(policy["verdict_thresholds"]),
    )
    coverage_complete = (
        not bm25_residual
        and not graph_residual
        and all(len(complete_qids[pool]) == 100 for pool in POOL_ORDER)
        and all(
            pool_inventory[pool]["candidate_rows"] == 400
            and pool_inventory[pool]["judged_candidate_rows"] == 400
            and pool_inventory[pool]["fully_judged_queries"] == 100
            for pool in POOL_ORDER
        )
    )
    if str(policy["version"]).endswith("-v3") and not coverage_complete:
        raise ValueError(
            "v3 requires five 400/400 judged pools, 100 complete queries per "
            "pool, and empty BM25/Graph residual manifests"
        )
    report = {
        "schema": "dense-semantic-drift-rescue-audit-report-v1",
        "version": str(policy["version"]),
        "status": (
            "COMPLETE_COVERAGE_COMPLETE"
            if coverage_complete
            else "COMPLETE_WITH_JUDGMENT_COVERAGE_LIMITS"
        ),
        "research_question": (
            "Can Deep-Dense, BM25, or strict native Graph rescue observable "
            "high-similarity/low-utility Dense Top-8 mismatch, and can one "
            "shared abstaining selector identify the rescue?"
        ),
        "frozen_dense_baseline": {
            "name": "SBERT D8",
            "queries": 100,
            "candidates_per_query": 8,
            "position_exact_replay_queries": 100,
            "reselected_for_this_audit": False,
        },
        "mismatch": {
            "candidate_counts": dict(sorted(mismatch_counts.items())),
            "query_counts": mismatch_query_counts,
            "drift_queries": sum(
                bool(drift_types[query_id] & {MISMATCH_A, MISMATCH_B})
                for query_id in qids
            ),
            "mean_drift_items_per_query": statistics.fmean(
                len(drift_items[query_id]) for query_id in qids
            ),
            "definitions": mismatch_cfg,
        },
        "candidate_pools": pool_inventory,
        "oracle": {
            "sources": oracle_summary,
            "matched_contrasts": oracle_contrasts,
            "unit": "query",
        },
        "realised_nested_oof": realised_summary,
        "route_aware_sensitivity": route_contrasts,
        "selector_contract": {
            "model": "HuberRegressor Direct Delta Regression",
            "NO_OP": True,
            "maximum_swaps_per_query": 1,
            "replacement_slots": (
                "all eight D8 positions; A/B/C labels are diagnostic only"
            ),
            "outer_splits": "exact report89 5 repeats × 5 query-grouped folds",
            "coverage_limited_pool_handling": (
                "filter exact split membership to fully judged four-candidate "
                "queries; never impute missing utility"
            ),
            "inner_splits": (
                "exact report89 deterministic inner membership, filtered only "
                "by pre-existing candidate judgment completeness"
            ),
            "kappa_options": policy["kappa_options"],
            "threshold_quantiles": policy["threshold_quantiles"],
            "source_blind_features": list(SOURCE_BLIND_FEATURES),
            "route_aware_additions": [
                name for name in ROUTE_AWARE_FEATURES
                if name not in SOURCE_BLIND_FEATURES
            ],
            "baseline_selector_score_difference_omitted": (
                "No frozen common selector score exists for BM25 candidates "
                "outside the original 1,600-pair pool; adding a missing proxy "
                "would encode source and violate the shared source-blind design."
            ),
            "gold_utility_used_as_inference_feature": False,
            "raw_reward_used_as_inference_feature": False,
            "source_identity_or_bonus_used": False,
        },
        "predeclared_verdicts": verdicts,
        "coverage_limits": {
            pool: {
                "fully_judged_queries": len(complete_qids[pool]),
                "of_100": 100,
            }
            for pool in POOL_ORDER
        },
        "assertions": {
            "D8_position_exact_queries": 100,
            "five_rescue_pools_400_of_400_judged": coverage_complete,
            "five_rescue_pools_complete_queries": {
                pool: len(complete_qids[pool]) for pool in POOL_ORDER
            },
            "bm25_residual_pairs": len(bm25_residual),
            "graph_practical_residual_pairs": len(graph_residual),
            "candidate_budget_every_pool_query": all(
                len(pool_ids[query_id][pool]) == budget
                for query_id in qids for pool in POOL_ORDER
            ),
            "pool_construction_used_utility": False,
            "missing_judgments_zero_filled": False,
            "realised_action_eligibility_used_mismatch_label": any(
                bool(row.get("eligibility_used_mismatch_label"))
                for row in oof_rows
            ),
            "query_grouped_split_unchanged": True,
            "test_read": False,
            "external_model_calls": 0,
            "strict_graph_exclusive_fallback_callback_padding": sum(
                bool(row["fallback_used"])
                or bool(row["callback_used"])
                or bool(row["padding_used"])
                for row in pool_rows
                if row["source_pool"] == POOL_GRAPH_EXCLUSIVE
            ),
            "source_blind_forbidden_features_present": [
                feature for feature in SOURCE_BLIND_FEATURES
                if any(
                    forbidden in feature.lower()
                    for forbidden in FORBIDDEN_INFERENCE_FEATURES
                )
            ],
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ))
    try:
        _write_json(
            temp_dir / "artefact_inventory.json",
            artefact_inventory,
        )
        _write_jsonl(
            temp_dir / "dense_mismatch_candidates.jsonl",
            dense_mismatch_rows,
        )
        _write_jsonl(
            temp_dir / "rescue_candidate_pools.jsonl",
            pool_rows,
        )
        _write_jsonl(
            temp_dir / "oracle_rescue_actions.jsonl",
            oracle_rows,
        )
        _write_jsonl(
            temp_dir / "oof_rescue_actions.jsonl",
            oof_rows,
        )
        _write_csv(temp_dir / "per_query_metrics.csv", per_query)
        _write_json(
            temp_dir / "route_rescue_summary.json",
            report,
        )
        _write_csv(
            temp_dir / "dimension_decomposition.csv",
            dimension_rows,
        )
        _write_jsonl(
            temp_dir / "residual_bm25_judgment_manifest.jsonl",
            bm25_residual,
        )
        _write_jsonl(
            temp_dir / "residual_graph_practical_judgment_manifest.jsonl",
            graph_residual,
        )
        _write_json(
            temp_dir / "nested_fold_audit.json",
            {
                "schema": "dense-drift-rescue-nested-fold-audit-v1",
                "source": str(paths["split_manifest"].relative_to(root)),
                "source_sha256": _sha256(paths["split_manifest"]),
                "folds": fold_audit,
            },
        )
        _render_similarity_utility_scatter(
            temp_dir / "similarity_utility_scatter.pdf",
            dense_rows=dense_mismatch_rows,
            pool_rows=pool_rows,
            oracle_rows=oracle_rows,
        )
        output_files = sorted(
            path for path in temp_dir.iterdir() if path.is_file()
        )
        manifest = {
            "schema": "dense-semantic-drift-rescue-audit-manifest-v1",
            "version": str(policy["version"]),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": report["status"],
            "queries": len(qids),
            "dense_candidate_rows": len(dense_mismatch_rows),
            "rescue_candidate_rows": len(pool_rows),
            "oracle_query_source_rows": len(oracle_rows),
            "oof_rows": len(oof_rows),
            "bm25_residual_pairs": len(bm25_residual),
            "graph_practical_residual_pairs": len(graph_residual),
            "triggered_verdicts": verdicts["triggered_verdicts"],
            "external_model_calls": 0,
            "test_read": False,
            "input_hashes": {
                str(path.relative_to(root)): _sha256(path)
                for path in required if path.is_file()
            },
            "output_hashes": {
                path.name: _sha256(path) for path in output_files
            },
        }
        _write_json(temp_dir / "manifest.json", manifest)
        os.replace(temp_dir, output_dir)
    except Exception:
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-key",
        default="dense_semantic_drift_rescue_audit",
    )
    args = parser.parse_args()
    report = run(args.config_key)
    print(json.dumps({
        "status": report["status"],
        "verdicts": report["predeclared_verdicts"]["triggered_verdicts"],
        "drift_queries": report["mismatch"]["drift_queries"],
        "output": project_config.params(
            args.config_key, "output_dir"
        ),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
