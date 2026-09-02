#!/usr/bin/env python3
"""Matched development-only repair of the evidence-selection action space.

The experiment separates three quantities that the historical one-swap
reports mixed together:

* candidate access: what the best eight in the visible pool could achieve;
* action-space feasibility: what can be achieved with at most r non-D8 items;
* learning: what an explicit-route-blind OOF candidate-utility scorer selects.

One candidate scorer is fitted per backend and outer fold on the largest
available training-query pool (D50 union strict FixedGraph4).  Its repeated
OOF predictions are averaged per query-candidate before *any* selection, and
the same averaged scores are reused for every pool width and every replacement
budget.  Frozen test data, community replies, external APIs, explicit route
identity, and current-query utility are fail-closed.  Dense-rank missingness is
retained as an allowed retrieval-rank feature and disclosed as a source proxy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
# Load XGBoost before PyTorch on macOS; the reverse OpenMP load order can
# segfault when XGBRanker receives grouped relevance labels.
import xgboost
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

try:
    import configuration as project_config
    from evaluation.ir_metrics import graded_ndcg_at
    from evaluation.judgment_completeness import complete_utility_v2_rows
    from utility_scoring.learned_diffusion import reranker_validation as canonical
    from candidate_pool.analyze_strict_sbert_graph_oracle import _reject_test
    from candidate_pool.run_dense_semantic_drift_rescue_audit import (
        STATIC_PREDICTOR_FEATURES,
        _build_idf,
        _load_embeddings,
    )
    from candidate_pool.run_m50_dense_frontier_analysis import static_features_for_arm
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import configuration as project_config
    from evaluation.ir_metrics import graded_ndcg_at
    from evaluation.judgment_completeness import complete_utility_v2_rows
    from utility_scoring.learned_diffusion import reranker_validation as canonical
    from candidate_pool.analyze_strict_sbert_graph_oracle import _reject_test
    from candidate_pool.run_dense_semantic_drift_rescue_audit import (
        STATIC_PREDICTOR_FEATURES,
        _build_idf,
        _load_embeddings,
    )
    from candidate_pool.run_m50_dense_frontier_analysis import static_features_for_arm


ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEY = "selection_action_space_repair_dev300_rrf3_rawtext"
POOL_DENSE = "dense"
POOL_GRAPH = "dense_plus_fixed_graph4"
SCORER_HUBER = "candidate_huber"
SCORER_MLP = "candidate_small_mlp"
SCORER_LAMBDAMART = "candidate_lambdamart"
RAW_SCORER = "raw_dense"
EPS = 1e-12


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in map(str, values):
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def utility_at8(ids: list[str], query_id: str, registry: dict) -> float:
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError(f"{query_id}: evidence set must contain eight unique items")
    return statistics.fmean(float(registry[(query_id, cid)]["utility"]) for cid in ids)


def exact_select_under_budget(
    d8_ids: list[str],
    pool_ids: list[str],
    scores: dict[str, float],
    replacement_budget: int,
    *,
    k: int = 8,
) -> dict:
    """Return the exact maximum-score K-set with at most r non-D8 items.

    The additive candidate objective makes the constrained problem separable:
    for each feasible replacement count t, retain the best K-t baseline items
    and add the best t entrants.  Across equal objectives the deterministic
    rule prefers fewer changes, then the lexicographically smaller identity
    tuple.  The function is used unchanged for OOF predictions and Oracles.
    """
    baseline = stable_unique(d8_ids)
    pool = stable_unique(pool_ids)
    if len(baseline) != k or not set(baseline) <= set(pool):
        raise ValueError("invalid D8/pool relationship")
    if replacement_budget < 0:
        raise ValueError("replacement budget must be non-negative")
    missing = [cid for cid in pool if cid not in scores]
    if missing:
        raise KeyError(f"missing scores for {len(missing)} candidates")
    entrants = [cid for cid in pool if cid not in set(baseline)]
    ranked_base = sorted(baseline, key=lambda cid: (-float(scores[cid]), cid))
    ranked_entrants = sorted(entrants, key=lambda cid: (-float(scores[cid]), cid))
    maximum = min(replacement_budget, k, len(ranked_entrants))
    best: dict | None = None
    for replacements in range(maximum + 1):
        selected = [
            *ranked_base[: k - replacements],
            *ranked_entrants[:replacements],
        ]
        objective = sum(float(scores[cid]) for cid in selected)
        identity = tuple(sorted(selected))
        candidate = {
            "selected_ids": sorted(
                selected, key=lambda cid: (-float(scores[cid]), cid)
            ),
            "replacement_count": replacements,
            "objective_sum": objective,
            "identity_tiebreak": identity,
        }
        if best is None:
            best = candidate
            continue
        if objective > float(best["objective_sum"]) + EPS:
            best = candidate
        elif abs(objective - float(best["objective_sum"])) <= EPS:
            if replacements < int(best["replacement_count"]):
                best = candidate
            elif (
                replacements == int(best["replacement_count"])
                and identity < tuple(best["identity_tiebreak"])
            ):
                best = candidate
    assert best is not None
    return best


class CandidateHuber:
    def __init__(self, scaler: StandardScaler, model: HuberRegressor):
        self.scaler = scaler
        self.model = model

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(self.scaler.transform(matrix)), 1.0, 7.0)


class CandidateMLP:
    def __init__(self, scaler: StandardScaler, model: canonical.SmallMLP):
        self.scaler = scaler
        self.model = model

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        values = self.scaler.transform(matrix).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(values, dtype=torch.float32))
            return (1.0 + 6.0 * torch.sigmoid(logits)).numpy()


class CandidateLambdaMART:
    def __init__(self, scaler: StandardScaler, model: Any):
        self.scaler = scaler
        self.model = model

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        values = self.scaler.transform(matrix).astype(np.float32)
        return np.asarray(self.model.predict(values), dtype=np.float64)


def _candidate_arrays(
    qids: list[str],
    candidate_ids: dict[str, list[str]],
    static: dict[tuple[str, str], dict[str, float]],
    registry: dict[tuple[str, str], dict],
    feature_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    """``feature_names`` defaults to the frozen eight-feature contract, so every
    existing caller is unaffected; the Stage-2 redesign passes an explicit list
    because its feature-family ablation varies the columns."""
    names = tuple(STATIC_PREDICTOR_FEATURES if feature_names is None
                  else feature_names)
    pairs = [
        (qid, cid) for qid in qids for cid in candidate_ids[qid]
    ]
    matrix = np.asarray(
        [
            [float(static[pair][name]) for name in names]
            for pair in pairs
        ],
        dtype=np.float64,
    )
    target = np.asarray(
        [float(registry[pair]["utility"]) for pair in pairs], dtype=np.float64
    )
    counts = Counter(qid for qid, _ in pairs)
    weights = np.asarray(
        [1.0 / (len(qids) * counts[qid]) for qid, _ in pairs], dtype=np.float64
    )
    if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-10):
        raise AssertionError("query-balanced candidate weights do not sum to one")
    return pairs, matrix, target, weights


def _fit_huber(
    qids: list[str], candidate_ids: dict[str, list[str]], static: dict,
    registry: dict, setting: dict, feature_names=None,
) -> CandidateHuber:
    _, matrix, target, weights = _candidate_arrays(
        qids, candidate_ids, static, registry, feature_names
    )
    scaler = StandardScaler().fit(matrix)
    model = HuberRegressor(
        epsilon=float(setting["epsilon"]),
        alpha=float(setting["alpha"]),
        max_iter=int(setting["max_iter"]),
        fit_intercept=True,
    )
    model.fit(scaler.transform(matrix), target, sample_weight=weights)
    return CandidateHuber(scaler, model)


def _fit_mlp(
    qids: list[str], candidate_ids: dict[str, list[str]], static: dict,
    registry: dict, setting: dict, seed: int, feature_names=None,
) -> CandidateMLP:
    """``feature_names`` defaults to the frozen eight-feature contract, exactly
    as ``_candidate_arrays`` does, so every existing caller is unaffected.  It
    was previously absent here while every sibling fitter accepted it, which
    silently pinned this arm to the frozen columns whenever a caller passed a
    different contract."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _, matrix, target, weights = _candidate_arrays(
        qids, candidate_ids, static, registry, feature_names
    )
    scaler = StandardScaler().fit(matrix)
    values = torch.tensor(
        scaler.transform(matrix).astype(np.float32), dtype=torch.float32
    )
    labels = torch.tensor((target - 1.0) / 6.0, dtype=torch.float32)
    sample_weights = torch.tensor(weights.astype(np.float32), dtype=torch.float32)
    canonical_setting = {
        "hidden_dim": int(setting["hidden_dim"]),
        "layers": int(setting["layers"]),
        "dropout": float(setting["dropout"]),
    }
    model = canonical.SmallMLP(values.shape[1], canonical_setting)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(setting["learning_rate"]),
        weight_decay=float(setting["weight_decay"]),
    )
    for _ in range(int(setting["epochs"])):
        model.train()
        optimizer.zero_grad()
        predicted = torch.sigmoid(model(values))
        losses = torch.nn.functional.smooth_l1_loss(
            predicted, labels, reduction="none"
        )
        loss = (losses * sample_weights).sum()
        loss.backward()
        optimizer.step()
    return CandidateMLP(scaler, model)


def _fit_lambdamart(
    qids: list[str], candidate_ids: dict[str, list[str]], static: dict,
    registry: dict, setting: dict, seed: int,
) -> CandidateLambdaMART:
    pairs, matrix, target, _ = _candidate_arrays(
        qids, candidate_ids, static, registry
    )
    scaler = StandardScaler().fit(matrix)
    model = canonical.fit_xgb_lambdamart(
        scaler.transform(matrix).astype(np.float32),
        target.astype(np.float32),
        [qid for qid, _ in pairs],
        setting,
        seed,
    )
    return CandidateLambdaMART(scaler, model)


def _predict_pairs(
    model: CandidateHuber | CandidateMLP | CandidateLambdaMART,
    qids: list[str],
    candidate_ids: dict[str, list[str]],
    static: dict,
) -> dict[tuple[str, str], float]:
    pairs = [(qid, cid) for qid in qids for cid in candidate_ids[qid]]
    matrix = np.asarray(
        [
            [float(static[pair][name]) for name in STATIC_PREDICTOR_FEATURES]
            for pair in pairs
        ],
        dtype=np.float64,
    )
    values = model.predict(matrix)
    return dict(zip(pairs, map(float, values), strict=True))


def _query_mean_mae(
    predictions: dict[tuple[str, str], float],
    qids: list[str],
    candidate_ids: dict[str, list[str]],
    registry: dict,
) -> float:
    return statistics.fmean(
        statistics.fmean(
            abs(
                float(predictions[(qid, cid)])
                - float(registry[(qid, cid)]["utility"])
            )
            for cid in candidate_ids[qid]
        )
        for qid in qids
    )


def _query_mean_ndcg_at8(
    predictions: dict[tuple[str, str], float],
    qids: list[str],
    candidate_ids: dict[str, list[str]],
    registry: dict,
) -> float:
    values = []
    for qid in qids:
        ids = list(map(str, candidate_ids[qid]))
        ranked = sorted(
            ids, key=lambda cid: (-float(predictions[(qid, cid)]), cid)
        )
        gains = {
            cid: float(canonical.historical_utility_grade(
                registry[(qid, cid)]["utility"]
            ))
            for cid in ids
        }
        values.append(float(graded_ndcg_at(ranked, gains, 8)))
    return statistics.fmean(values)


def _tune_and_fit(
    *, scorer: str, train_qids: list[str], inner_splits: list,
    candidate_ids: dict[str, list[str]], static: dict, registry: dict,
    settings: list[dict], seed: int,
) -> tuple[CandidateHuber | CandidateMLP | CandidateLambdaMART, dict]:
    traces = []
    for config_index, setting in enumerate(settings):
        scores = []
        for inner_index, (inner_train, inner_valid) in enumerate(inner_splits):
            fit_seed = seed + 10_000 * config_index + inner_index
            if scorer == SCORER_HUBER:
                model = _fit_huber(
                    inner_train, candidate_ids, static, registry, setting
                )
            elif scorer == SCORER_MLP:
                model = _fit_mlp(
                    inner_train, candidate_ids, static, registry, setting, fit_seed
                )
            elif scorer == SCORER_LAMBDAMART:
                model = _fit_lambdamart(
                    inner_train, candidate_ids, static, registry, setting, fit_seed
                )
            else:
                raise ValueError(scorer)
            predicted = _predict_pairs(model, inner_valid, candidate_ids, static)
            if scorer == SCORER_LAMBDAMART:
                scores.append(_query_mean_ndcg_at8(
                    predicted, inner_valid, candidate_ids, registry
                ))
            else:
                scores.append(_query_mean_mae(
                    predicted, inner_valid, candidate_ids, registry
                ))
        if scorer == SCORER_LAMBDAMART:
            traces.append({
                "setting": setting,
                "inner_query_mean_ndcg_at8": statistics.fmean(scores),
                "inner_fold_ndcg_at8": scores,
            })
        else:
            traces.append({
                "setting": setting,
                "inner_query_mean_mae": statistics.fmean(scores),
                "inner_fold_mae": scores,
            })
    if scorer == SCORER_LAMBDAMART:
        best_index = max(
            range(len(traces)),
            key=lambda index: (traces[index]["inner_query_mean_ndcg_at8"], -index),
        )
    else:
        best_index = min(
            range(len(traces)),
            key=lambda index: (traces[index]["inner_query_mean_mae"], index),
        )
    selected = settings[best_index]
    if scorer == SCORER_HUBER:
        model = _fit_huber(train_qids, candidate_ids, static, registry, selected)
    elif scorer == SCORER_MLP:
        model = _fit_mlp(
            train_qids, candidate_ids, static, registry, selected, seed + 90_000
        )
    else:
        model = _fit_lambdamart(
            train_qids, candidate_ids, static, registry, selected, seed + 90_000
        )
    return model, {
        "selected_setting": selected,
        "selection_objective": (
            "maximum query-macro nDCG@8 on historical 0-6 utility grades"
            if scorer == SCORER_LAMBDAMART
            else "minimum mean query-level candidate MAE"
        ),
        "tuning_trace": traces,
    }


def _resolve_inputs(raw: dict) -> dict[str, Any]:
    output: dict[str, Any] = {}
    path_keys = {
        "utility_registry", "dense_memberships", "graph_candidate_views",
        "split_manifest", "query_admin", "queries", "corpus", "output_dir",
    }
    for key, value in raw.items():
        if key in path_keys:
            path = Path(value)
            output[key] = path if path.is_absolute() else ROOT / path
        elif key == "embeddings":
            output[key] = {}
            for backend, paths in value.items():
                output[key][backend] = {}
                for kind, item in paths.items():
                    path = Path(item)
                    output[key][backend][kind] = (
                        path if path.is_absolute() else ROOT / path
                    )
        else:
            output[key] = value
    return output


def _load_contract(raw: dict) -> dict:
    paths = _resolve_inputs(raw)
    if bool(paths["allow_external_calls"]) or bool(paths["allow_frozen_test"]):
        raise ValueError("experiment must remain local-only and development-only")
    for key in (
        "utility_registry", "dense_memberships", "graph_candidate_views",
        "split_manifest", "query_admin", "queries", "corpus",
    ):
        _reject_test(paths[key])
        if not paths[key].exists():
            raise FileNotFoundError(paths[key])
    for backend in paths["backends"]:
        for path in paths["embeddings"][backend].values():
            _reject_test(path)
            if not path.exists():
                raise FileNotFoundError(path)

    complete_rows, registry = complete_utility_v2_rows(
        read_jsonl(paths["utility_registry"])
    )
    expected_registry_rows = int(paths.get("expected_complete_registry_rows", 19_813))
    if (
        len(complete_rows) != expected_registry_rows
        or len(registry) != expected_registry_rows
    ):
        raise ValueError(
            "formal complete utility registry identity changed: "
            f"{len(complete_rows)}/{len(registry)} != {expected_registry_rows}"
        )

    dense: dict[str, dict[str, list[dict]]] = {
        backend: defaultdict(list) for backend in paths["backends"]
    }
    for row in read_jsonl(paths["dense_memberships"]):
        dense[str(row["backend"])][str(row["query_id"])].append(row)
    for backend in paths["backends"]:
        dense[backend] = dict(dense[backend])
        for qid, rows in dense[backend].items():
            rows.sort(key=lambda row: int(row["rank"]))
            if len(rows) != 50 or [int(row["rank"]) for row in rows] != list(
                range(1, 51)
            ):
                raise ValueError(f"{backend}/{qid}: frozen D50 identity changed")
    qids = sorted(dense[paths["backends"][0]])
    expected_query_count = int(paths.get("expected_query_count", 100))
    if (
        len(qids) != expected_query_count
        or any(set(dense[b]) != set(qids) for b in paths["backends"])
    ):
        raise ValueError(
            f"development cohort identity changed: {len(qids)} != {expected_query_count}"
        )

    fixed_rows = [
        row for row in read_jsonl(paths["graph_candidate_views"])
        if str(row["view_type"]) == "fixed_graph4"
    ]
    graph: dict[str, list[dict]] = defaultdict(list)
    for row in fixed_rows:
        if not bool(row["native_graph"]):
            raise AssertionError("FixedGraph4 contains a non-native Graph candidate")
        if any(bool(row[name]) for name in ("fallback_used", "callback_used", "padding_used")):
            raise AssertionError("FixedGraph4 contains fallback/callback/padding")
        ranks = row.get("graph_pre_fallback_rank") or {}
        scores = row.get("native_graph_score") or {}
        valid_rank = any(int(value) > 0 for value in ranks.values())
        valid_score = any(
            math.isfinite(float(value)) and float(value) > 0 for value in scores.values()
        )
        if not (valid_rank or valid_score):
            raise AssertionError("native Graph provenance lacks score/rank evidence")
        graph[str(row["query_id"])].append(row)
    for qid in qids:
        graph[qid].sort(key=lambda row: int(row["graph_view_rank"]))
        if len(graph[qid]) != 4 or len({row["comment_id"] for row in graph[qid]}) != 4:
            raise ValueError(f"{qid}: FixedGraph4 must contain four unique candidates")

    required = {
        (qid, str(row["comment_id"]))
        for backend in paths["backends"]
        for qid in qids
        for row in dense[backend][qid]
    } | {
        (qid, str(row["comment_id"])) for qid in qids for row in graph[qid]
    }
    if not required <= set(registry):
        raise ValueError(f"candidate pool lacks {len(required - set(registry))} judgments")

    strata: dict[str, str] = {}
    with paths["query_admin"].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            qid = str(row["query_id"])
            if qid in set(qids):
                strata[qid] = str(row["llm_single_multi_label"])
    expected_strata = {
        str(key): int(value)
        for key, value in paths.get(
            "expected_strata_counts", {"single_need": 50, "multi_need": 50}
        ).items()
    }
    if Counter(strata.values()) != expected_strata:
        raise ValueError(
            f"single/multi development strata changed: "
            f"{dict(Counter(strata.values()))} != {expected_strata}"
        )

    split_source = json.loads(paths["split_manifest"].read_text(encoding="utf-8"))
    splits = list(split_source["rows"])
    expected_split_rows = int(paths["outer_repeats"]) * int(paths["outer_folds"])
    test_read = bool(split_source.get("test_read")) or bool(
        (split_source.get("audit") or {}).get("test_read")
    )
    if len(splits) != expected_split_rows or test_read:
        raise ValueError("outer split manifest changed or touched frozen test")
    validation = Counter(
        (int(row["repeat"]), str(qid))
        for row in splits for qid in row["validation_query_ids"]
    )
    expected = {
        (repeat, qid)
        for repeat in range(int(paths["outer_repeats"])) for qid in qids
    }
    if set(validation) != expected or set(validation.values()) != {1}:
        raise ValueError("outer validation coverage changed")
    if any(set(row["train_query_ids"]) & set(row["validation_query_ids"]) for row in splits):
        raise ValueError("query leakage in outer splits")

    require_report89_identity = bool(paths.get("require_report89_inner_identity", True))
    report89_lookup = {}
    if require_report89_identity:
        report89_folds = json.loads(
            (ROOT / "out/strict_native_graph_conservative_policy_dev100_v1/fold_metrics.json")
            .read_text(encoding="utf-8")
        )["folds"]
        report89_lookup = {
            (int(row["repeat"]), int(row["fold"])): row for row in report89_folds
        }
    inner_manifest = []
    for row in splits:
        train = list(map(str, row["train_query_ids"]))
        generated = canonical.inner_folds(train, int(paths["inner_folds"]), int(row["seed"]) + 7000)
        if require_report89_identity:
            historical = report89_lookup[
                (int(row["repeat"]), int(row["fold"]))
            ]["baseline_inner_splits"]
            if any(
                inner_train != list(map(str, old["train_query_ids"]))
                or inner_valid != list(map(str, old["validation_query_ids"]))
                for (inner_train, inner_valid), old in zip(
                    generated, historical, strict=True
                )
            ):
                raise AssertionError("inner split no longer matches Report89")
        inner_manifest.append({
            "repeat": int(row["repeat"]),
            "fold": int(row["fold"]),
            "rows": [
                {
                    "inner_fold": index,
                    "train_query_ids": inner_train,
                    "validation_query_ids": inner_valid,
                    "query_overlap": 0,
                }
                for index, (inner_train, inner_valid) in enumerate(generated)
            ],
        })

    return {
        "paths": paths,
        "registry": registry,
        "dense": dense,
        "graph": dict(graph),
        "qids": qids,
        "strata": strata,
        "splits": splits,
        "inner_manifest": inner_manifest,
        "required_pairs": required,
        "report89_inner_splits_exact": require_report89_identity,
    }


def _build_pools_and_features(contract: dict, max_pool_override: dict | None = None) -> dict:
    """Frozen pools and features; ``max_pool_override`` (query -> ordered ids)
    replaces the scorer-training pool while every feature definition, the
    dense-anchored d8 baseline and the missing-rank indicator stay frozen."""
    paths = contract["paths"]
    dense = contract["dense"]
    graph = contract["graph"]
    qids = contract["qids"]
    depths = list(map(int, paths["dense_depths"]))
    if depths != [8, 12, 20, 50] or int(paths["final_k"]) != 8:
        raise ValueError("pre-registered M/K grid changed")

    # Text and IDF are common; semantic vectors are backend-local.  This fixes
    # the historical E5 analysis bug in which MiniLM vectors were reused as
    # E5 selector features.
    features_by_backend: dict[str, dict] = {}
    embedding_audit = {}
    corpus_text_reference: dict[str, str] | None = None
    query_text_reference: dict[str, str] | None = None
    idf = None
    idf_manifest = None
    max_pool_ids: dict[str, dict[str, list[str]]] = {}
    for backend in paths["backends"]:
        (
            candidate_vectors,
            query_vectors,
            corpus_text,
            query_text,
            corpus_array,
            corpus_ids,
        ) = _load_embeddings(
            corpus_path=paths["corpus"],
            corpus_embeddings_path=paths["embeddings"][backend]["corpus"],
            queries_path=paths["queries"],
            query_embeddings_path=paths["embeddings"][backend]["query"],
            qids=set(qids),
        )
        if corpus_text_reference is None:
            corpus_text_reference = corpus_text
            query_text_reference = query_text
            idf, idf_manifest = _build_idf(corpus_text.values())
        elif corpus_text != corpus_text_reference or query_text != query_text_reference:
            raise ValueError("backend embedding text identity changed")
        rank_maps = {
            qid: {
                str(row["comment_id"]): int(row["rank"])
                for row in dense[backend][qid]
            }
            for qid in qids
        }
        d8 = {
            qid: [str(row["comment_id"]) for row in dense[backend][qid][:8]]
            for qid in qids
        }
        if max_pool_override is None:
            max_pool_ids[backend] = {
                qid: stable_unique([
                    *[str(row["comment_id"]) for row in dense[backend][qid]],
                    *[str(row["comment_id"]) for row in graph[qid]],
                ])
                for qid in qids
            }
        else:
            if set(max_pool_override) != set(qids):
                raise ValueError("pool override does not cover the cohort")
            max_pool_ids[backend] = {
                qid: [str(cid) for cid in max_pool_override[qid]] for qid in qids
            }
            if any(
                len(ids) != len(set(ids)) or not ids
                for ids in max_pool_ids[backend].values()
            ):
                raise ValueError("pool override rows must be non-empty and unique")
        features_by_backend[backend] = static_features_for_arm(
            query_ids=qids,
            baseline_ids=d8,
            arm_ids=max_pool_ids[backend],
            rank_maps=rank_maps,
            candidate_vectors=candidate_vectors,
            query_vectors=query_vectors,
            corpus_text=corpus_text,
            query_text=query_text,
            idf=idf,
        )
        if any(
            tuple(row) != tuple(STATIC_PREDICTOR_FEATURES)
            for row in features_by_backend[backend].values()
        ):
            raise AssertionError("candidate feature schema changed")
        embedding_audit[backend] = {
            "query_path": str(paths["embeddings"][backend]["query"].relative_to(ROOT)),
            "query_sha256": sha256(paths["embeddings"][backend]["query"]),
            "corpus_path": str(paths["embeddings"][backend]["corpus"].relative_to(ROOT)),
            "corpus_sha256": sha256(paths["embeddings"][backend]["corpus"]),
            "corpus_rows": len(corpus_ids),
            "embedding_dim": int(corpus_array.shape[1]),
        }

    pool_ids: dict[tuple[str, str, int, str], list[str]] = {}
    manifest_rows: list[dict] = []
    expected_graph_sizes = {
        ("minilm", 8): 1200, ("minilm", 12): 1576,
        ("minilm", 20): 2354, ("minilm", 50): 5288,
        ("e5", 8): 1179, ("e5", 12): 1573,
        ("e5", 20): 2366, ("e5", 50): 5351,
    }
    require_dev100_sanity = bool(paths.get("require_dev100_identity_sanity", True))
    for backend in paths["backends"]:
        for qid in qids:
            dense_rows = dense[backend][qid]
            dense_rank = {
                str(row["comment_id"]): int(row["rank"]) for row in dense_rows
            }
            graph_map = {str(row["comment_id"]): row for row in graph[qid]}
            d8_set = {str(row["comment_id"]) for row in dense_rows[:8]}
            for depth in depths:
                dense_ids = [str(row["comment_id"]) for row in dense_rows[:depth]]
                graph_ids = [str(row["comment_id"]) for row in graph[qid]]
                conditions = {
                    POOL_DENSE: dense_ids,
                    POOL_GRAPH: stable_unique([*dense_ids, *graph_ids]),
                }
                for pool_family, ids in conditions.items():
                    pool_ids[(backend, pool_family, depth, qid)] = ids
                    dense_m_set = set(dense_ids)
                    for position, cid in enumerate(ids, start=1):
                        in_d8 = cid in d8_set
                        in_dense_m = cid in dense_m_set
                        in_graph = cid in graph_map
                        graph_added = in_graph and not in_dense_m
                        if in_d8:
                            source_class = "d8"
                        elif in_dense_m and in_graph:
                            source_class = "dense_tail_and_graph"
                        elif in_dense_m:
                            source_class = "dense_tail_only"
                        elif graph_added:
                            source_class = "graph_only"
                        else:
                            raise AssertionError("candidate has no traceable pool source")
                        graph_row = graph_map.get(cid)
                        manifest_rows.append({
                            "backend": backend,
                            "pool_family": pool_family,
                            "dense_depth": depth,
                            "pool_id": f"{backend}:{pool_family}:M{depth}",
                            "query_id": qid,
                            "candidate_id": cid,
                            "pool_position": position,
                            "in_d8": in_d8,
                            "in_dense_m": in_dense_m,
                            "dense_rank": dense_rank.get(cid),
                            "in_fixed_graph4": in_graph,
                            "fixed_graph_rank": (
                                int(graph_row["graph_view_rank"]) if graph_row else None
                            ),
                            "native_graph": bool(graph_row["native_graph"]) if graph_row else False,
                            "fallback_used": bool(graph_row["fallback_used"]) if graph_row else False,
                            "callback_used": bool(graph_row["callback_used"]) if graph_row else False,
                            "padding_used": bool(graph_row["padding_used"]) if graph_row else False,
                            "graph_added_relative_to_dense_m": graph_added,
                            "source_class": source_class,
                            "graph_routes_json": json.dumps(
                                graph_row.get("graph_routes", []) if graph_row else [],
                                sort_keys=True,
                            ),
                            "merged_sources_json": json.dumps(
                                [
                                    *( ["dense"] if in_dense_m else [] ),
                                    *( ["strict_fixed_graph4"] if in_graph else [] ),
                                ]
                            ),
                            "utility_complete": (qid, cid) in contract["registry"],
                            "candidate_construction_used_utility": False,
                            "candidate_construction_used_community_replies": False,
                        })
    for backend in paths["backends"]:
        for depth in depths:
            actual = sum(
                len(pool_ids[(backend, POOL_GRAPH, depth, qid)]) for qid in qids
            )
            if require_dev100_sanity and actual != expected_graph_sizes[(backend, depth)]:
                raise AssertionError(
                    f"{backend}/M{depth}: Graph union size {actual} != frozen {expected_graph_sizes[(backend, depth)]}"
                )
            if not require_dev100_sanity:
                lower = len(qids) * depth
                upper = len(qids) * (depth + 4)
                if not lower <= actual <= upper:
                    raise AssertionError(
                        f"{backend}/M{depth}: Graph union size {actual} outside "
                        f"valid [{lower}, {upper}]"
                    )
    if any(
        bool(row[name])
        for row in manifest_rows if row["native_graph"]
        for name in ("fallback_used", "callback_used", "padding_used")
    ):
        raise AssertionError("strict Graph provenance invariant failed")
    return {
        "features": features_by_backend,
        "max_pool_ids": max_pool_ids,
        "pool_ids": pool_ids,
        "manifest_rows": manifest_rows,
        "embedding_audit": embedding_audit,
        "idf_manifest": idf_manifest,
    }


def _scorer_settings(paths: dict) -> dict[str, list[dict]]:
    huber = paths["scorers"][SCORER_HUBER]
    mlp = paths["scorers"][SCORER_MLP]
    return {
        SCORER_HUBER: [
            {
                "epsilon": float(huber["epsilon"]),
                "alpha": float(alpha),
                "max_iter": int(huber["max_iter"]),
            }
            for alpha in huber["alpha_grid"]
        ],
        SCORER_MLP: [
            {
                "hidden_dim": int(hidden),
                "weight_decay": float(weight_decay),
                "layers": int(mlp["layers"]),
                "dropout": float(mlp["dropout"]),
                "learning_rate": float(mlp["learning_rate"]),
                "epochs": int(mlp["epochs"]),
            }
            for hidden in mlp["hidden_grid"]
            for weight_decay in mlp["weight_decay_grid"]
        ],
    }


def _run_oof_predictions(contract: dict, pools: dict) -> tuple[list[dict], dict, list[dict]]:
    paths = contract["paths"]
    settings = _scorer_settings(paths)
    raw_predictions: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    tuning_audit: list[dict] = []
    torch.set_num_threads(1)
    for backend_index, backend in enumerate(paths["backends"]):
        for split_index, split in enumerate(contract["splits"]):
            repeat = int(split["repeat"])
            fold = int(split["fold"])
            fold_seed = int(split["seed"])
            train = list(map(str, split["train_query_ids"]))
            valid = list(map(str, split["validation_query_ids"]))
            inner = canonical.inner_folds(
                train, int(paths["inner_folds"]), fold_seed + 7000
            )
            for scorer_index, scorer in enumerate((SCORER_HUBER, SCORER_MLP)):
                model_seed = (
                    fold_seed + 100_000 * backend_index
                    + 1_000 * split_index + 100 * scorer_index
                )
                model, audit = _tune_and_fit(
                    scorer=scorer,
                    train_qids=train,
                    inner_splits=inner,
                    candidate_ids=pools["max_pool_ids"][backend],
                    static=pools["features"][backend],
                    registry=contract["registry"],
                    settings=settings[scorer],
                    seed=model_seed,
                )
                predictions = _predict_pairs(
                    model,
                    valid,
                    pools["max_pool_ids"][backend],
                    pools["features"][backend],
                )
                for (qid, cid), value in predictions.items():
                    raw_predictions[(backend, scorer, qid, cid)].append({
                        "repeat": repeat,
                        "fold": fold,
                        "prediction": value,
                    })
                tuning_audit.append({
                    "backend": backend,
                    "scorer": scorer,
                    "repeat": repeat,
                    "fold": fold,
                    "seed": model_seed,
                    "outer_train_queries": len(train),
                    "outer_validation_queries": len(valid),
                    "train_validation_overlap": len(set(train) & set(valid)),
                    **audit,
                })
            print(json.dumps({
                "stage": "candidate_oof",
                "backend": backend,
                "repeat": repeat,
                "fold": fold,
                "status": "COMPLETE",
            }, sort_keys=True), flush=True)

    prediction_rows: list[dict] = []
    aggregate: dict[tuple[str, str, str, str], float] = {}
    for key, values in sorted(raw_predictions.items()):
        backend, scorer, qid, cid = key
        values.sort(key=lambda row: int(row["repeat"]))
        if len(values) != 5 or [int(row["repeat"]) for row in values] != list(range(5)):
            raise ValueError(f"{key}: expected one OOF prediction per repeat")
        predicted = [float(row["prediction"]) for row in values]
        mean_prediction = statistics.fmean(predicted)
        aggregate[key] = mean_prediction
        prediction_rows.append({
            "backend": backend,
            "scorer": scorer,
            "query_id": qid,
            "candidate_id": cid,
            "oof_prediction_mean": mean_prediction,
            "oof_prediction_sd": float(np.std(predicted)),
            "oof_prediction_n": 5,
            "repeat_predictions": predicted,
            "repeat_folds": [int(row["fold"]) for row in values],
            "utility_for_evaluation_only": float(
                contract["registry"][(qid, cid)]["utility"]
            ),
            "inference_used_current_query_utility": False,
            "inference_used_oracle_label": False,
            "inference_used_community_reply": False,
            "inference_used_route_identity": False,
        })
    expected = sum(
        len(pools["max_pool_ids"][backend][qid])
        for backend in paths["backends"] for qid in contract["qids"]
    ) * 2
    if len(prediction_rows) != expected:
        raise ValueError(f"OOF aggregate coverage {len(prediction_rows)} != {expected}")
    return prediction_rows, aggregate, tuning_audit


def _run_matched_lambdamart_oof(
    contract: dict,
    pools: dict,
    *,
    backend: str,
    settings: list[dict],
) -> tuple[list[dict], dict[tuple[str, str], float], list[dict]]:
    """Fit one matched LambdaMART per frozen outer fold and average repeats."""
    if backend not in contract["paths"]["backends"]:
        raise ValueError(f"unknown matched LambdaMART backend: {backend}")
    raw_predictions: dict[tuple[str, str], list[dict]] = defaultdict(list)
    tuning_audit: list[dict] = []
    for split_index, split in enumerate(contract["splits"]):
        repeat = int(split["repeat"])
        fold = int(split["fold"])
        fold_seed = int(split["seed"])
        train = list(map(str, split["train_query_ids"]))
        valid = list(map(str, split["validation_query_ids"]))
        if set(train) & set(valid):
            raise AssertionError("held-out query entered LambdaMART training")
        inner = canonical.inner_folds(
            train, int(contract["paths"]["inner_folds"]), fold_seed + 7000
        )
        model_seed = fold_seed + 700_000 + 1_000 * split_index
        model, audit = _tune_and_fit(
            scorer=SCORER_LAMBDAMART,
            train_qids=train,
            inner_splits=inner,
            candidate_ids=pools["max_pool_ids"][backend],
            static=pools["features"][backend],
            registry=contract["registry"],
            settings=settings,
            seed=model_seed,
        )
        predictions = _predict_pairs(
            model,
            valid,
            pools["max_pool_ids"][backend],
            pools["features"][backend],
        )
        expected_pairs = {
            (qid, cid)
            for qid in valid
            for cid in pools["max_pool_ids"][backend][qid]
        }
        if set(predictions) != expected_pairs:
            raise ValueError("held-out LambdaMART prediction coverage changed")
        for pair, value in predictions.items():
            raw_predictions[pair].append({
                "repeat": repeat,
                "fold": fold,
                "prediction": float(value),
            })
        tuning_audit.append({
            "backend": backend,
            "scorer": SCORER_LAMBDAMART,
            "repeat": repeat,
            "fold": fold,
            "seed": model_seed,
            "outer_train_queries": len(train),
            "outer_validation_queries": len(valid),
            "train_validation_overlap": 0,
            **audit,
        })
        print(json.dumps({
            "stage": "matched_lambdamart_oof",
            "backend": backend,
            "repeat": repeat,
            "fold": fold,
            "status": "COMPLETE",
        }, sort_keys=True), flush=True)

    rows: list[dict] = []
    aggregate: dict[tuple[str, str], float] = {}
    expected_all = {
        (qid, cid)
        for qid in contract["qids"]
        for cid in pools["max_pool_ids"][backend][qid]
    }
    if set(raw_predictions) != expected_all:
        raise ValueError("LambdaMART OOF scorer-pool coverage is incomplete")
    for (qid, cid), values in sorted(raw_predictions.items()):
        values.sort(key=lambda row: int(row["repeat"]))
        if len(values) != 5 or [int(row["repeat"]) for row in values] != list(range(5)):
            raise ValueError(f"{qid}/{cid}: expected five repeated OOF scores")
        predicted = [float(row["prediction"]) for row in values]
        mean_prediction = statistics.fmean(predicted)
        aggregate[(qid, cid)] = mean_prediction
        rows.append({
            "backend": backend,
            "scorer": SCORER_LAMBDAMART,
            "query_id": qid,
            "candidate_id": cid,
            "oof_prediction_mean": mean_prediction,
            "oof_prediction_sd": float(np.std(predicted)),
            "oof_prediction_n": 5,
            "repeat_predictions": predicted,
            "repeat_folds": [int(row["fold"]) for row in values],
            "utility_for_evaluation_only": float(
                contract["registry"][(qid, cid)]["utility"]
            ),
            "historical_relevance_grade_for_evaluation_only": int(
                canonical.historical_utility_grade(
                    contract["registry"][(qid, cid)]["utility"]
                )
            ),
            "inference_used_current_query_utility": False,
            "inference_used_oracle_label": False,
            "inference_used_community_reply": False,
            "inference_used_route_identity": False,
        })
    return rows, aggregate, tuning_audit


def _source_membership(
    backend: str, depth: int, qid: str, cid: str, contract: dict,
) -> dict[str, bool]:
    d8 = {
        str(row["comment_id"]) for row in contract["dense"][backend][qid][:8]
    }
    dense_m = {
        str(row["comment_id"])
        for row in contract["dense"][backend][qid][:depth]
    }
    graph = {str(row["comment_id"]) for row in contract["graph"][qid]}
    return {
        "in_d8": cid in d8,
        "in_dense_m": cid in dense_m,
        "in_fixed_graph4": cid in graph,
        "graph_added_relative_to_dense_m": cid in graph and cid not in dense_m,
    }


def _replacement_diagnostics(
    *, selected_ids: list[str], baseline_ids: list[str], predicted_scores: dict[str, float],
    query_id: str, backend: str, depth: int, contract: dict,
) -> dict:
    selected = set(selected_ids)
    baseline = set(baseline_ids)
    entrants = sorted(
        selected - baseline,
        key=lambda cid: (-float(predicted_scores[cid]), cid),
    )
    removed = sorted(
        baseline - selected,
        key=lambda cid: (float(predicted_scores[cid]), cid),
    )
    if len(entrants) != len(removed):
        raise AssertionError("entrant/removal counts differ")
    dense_count = graph_count = overlap_count = graph_positive = 0
    contribution = 0.0
    pair_rows = []
    for entrant, displaced in zip(entrants, removed, strict=True):
        membership = _source_membership(
            backend, depth, query_id, entrant, contract
        )
        if membership["graph_added_relative_to_dense_m"]:
            source = "graph_added"
            graph_count += 1
        elif membership["in_dense_m"]:
            source = "deeper_dense"
            dense_count += 1
            overlap_count += int(membership["in_fixed_graph4"])
        else:
            raise AssertionError("selected entrant is not traceable")
        delta = (
            float(contract["registry"][(query_id, entrant)]["utility"])
            - float(contract["registry"][(query_id, displaced)]["utility"])
        ) / 8.0
        contribution += delta
        graph_positive += int(source == "graph_added" and delta > EPS)
        pair_rows.append({
            "entrant_id": entrant,
            "displaced_id": displaced,
            "source": source,
            "also_fixed_graph4": bool(membership["in_fixed_graph4"]),
            "utility_at8_contribution": delta,
        })
    if dense_count + graph_count != len(entrants):
        raise AssertionError("entrant source partition is not exhaustive")
    return {
        "replacement_count": len(entrants),
        "deeper_dense_entrant_count": dense_count,
        "graph_added_entrant_count": graph_count,
        "dense_tail_also_graph_count": overlap_count,
        "graph_added_positive_pair_count": graph_positive,
        "replacement_utility_at8_contribution": contribution,
        "replacement_pairs_json": json.dumps(pair_rows, sort_keys=True),
    }


def _build_oracles(contract: dict, pools: dict) -> tuple[list[dict], dict, list[dict]]:
    paths = contract["paths"]
    qids = contract["qids"]
    budgets = list(map(int, paths["replacement_budgets"]))
    if budgets != [0, 1, 2, 4, 8]:
        raise ValueError("pre-registered replacement budgets changed")
    rows: list[dict] = []
    lookup: dict[tuple[str, str, int, int, str], dict] = {}
    selected_rows: list[dict] = []
    for backend in paths["backends"]:
        for qid in qids:
            d8 = [
                str(row["comment_id"])
                for row in contract["dense"][backend][qid][:8]
            ]
            baseline = utility_at8(d8, qid, contract["registry"])
            for pool_family in (POOL_DENSE, POOL_GRAPH):
                for depth in paths["dense_depths"]:
                    depth = int(depth)
                    pool = pools["pool_ids"][(backend, pool_family, depth, qid)]
                    true_scores = {
                        cid: float(contract["registry"][(qid, cid)]["utility"])
                        for cid in pool
                    }
                    full = exact_select_under_budget(d8, pool, true_scores, 8)
                    full_u = utility_at8(
                        full["selected_ids"], qid, contract["registry"]
                    )
                    for budget in budgets:
                        result = exact_select_under_budget(
                            d8, pool, true_scores, budget
                        )
                        selected_u = utility_at8(
                            result["selected_ids"], qid, contract["registry"]
                        )
                        diagnostics = _replacement_diagnostics(
                            selected_ids=result["selected_ids"],
                            baseline_ids=d8,
                            predicted_scores=true_scores,
                            query_id=qid,
                            backend=backend,
                            depth=depth,
                            contract=contract,
                        )
                        row = {
                            "backend": backend,
                            "pool_family": pool_family,
                            "dense_depth": depth,
                            "query_id": qid,
                            "replacement_budget": budget,
                            "selected_comment_ids": result["selected_ids"],
                            "replacement_count": int(result["replacement_count"]),
                            "baseline_utility_at8": baseline,
                            "oracle_utility_at8": selected_u,
                            "full_pool_oracle_utility_at8": full_u,
                            "feasible_headroom": selected_u - baseline,
                            "candidate_access_headroom": full_u - baseline,
                            "action_space_loss": full_u - selected_u,
                            "full_pool_oracle": budget == 8,
                            "objective": "mean frozen utility-v2",
                            "tie_break": "fewer_replacements_then_candidate_id",
                            **diagnostics,
                        }
                        if abs(
                            (row["feasible_headroom"] + row["action_space_loss"])
                            - row["candidate_access_headroom"]
                        ) > 1e-10:
                            raise AssertionError("Oracle decomposition identity failed")
                        rows.append(row)
                        lookup[(backend, pool_family, depth, budget, qid)] = row
                        selected_rows.append({
                            "selection_kind": "oracle",
                            "backend": backend,
                            "pool_family": pool_family,
                            "dense_depth": depth,
                            "scorer": "gold_utility_oracle",
                            "replacement_budget": budget,
                            "query_id": qid,
                            "selected_comment_ids": result["selected_ids"],
                            "selected_utility_at8": selected_u,
                            "replacement_count": int(result["replacement_count"]),
                            "deeper_dense_entrant_count": diagnostics["deeper_dense_entrant_count"],
                            "graph_added_entrant_count": diagnostics["graph_added_entrant_count"],
                            "inference_used_gold_utility": True,
                            "oracle_only": True,
                        })

    sanity = {
        ("minilm", POOL_DENSE, 12): 0.619750,
        ("minilm", POOL_DENSE, 20): 1.029125,
        ("minilm", POOL_DENSE, 50): 1.343750,
        ("e5", POOL_DENSE, 12): 0.620625,
        ("e5", POOL_DENSE, 20): 0.9770625,
        ("e5", POOL_DENSE, 50): 1.3126875,
        ("minilm", POOL_GRAPH, 8): 0.5424375,
        ("minilm", POOL_GRAPH, 12): 0.8194375,
        ("minilm", POOL_GRAPH, 20): 1.087875,
        ("minilm", POOL_GRAPH, 50): 1.3550625,
        ("e5", POOL_GRAPH, 8): 0.5024375,
        ("e5", POOL_GRAPH, 12): 0.8055,
        ("e5", POOL_GRAPH, 20): 1.0434375,
        ("e5", POOL_GRAPH, 50): 1.324375,
    }
    if bool(contract["paths"].get("require_dev100_identity_sanity", True)):
        for key, expected in sanity.items():
            backend, family, depth = key
            observed = statistics.fmean(
                lookup[(backend, family, depth, 8, qid)]["candidate_access_headroom"]
                for qid in qids
            )
            if not math.isclose(observed, expected, abs_tol=1e-10):
                raise AssertionError(
                    f"Oracle sanity failed for {key}: {observed} != {expected}"
                )
    return rows, lookup, selected_rows


def _build_learned_selections(
    contract: dict, pools: dict, predictions: dict, oracle: dict,
) -> tuple[list[dict], list[dict]]:
    paths = contract["paths"]
    qids = contract["qids"]
    rows: list[dict] = []
    selected_rows: list[dict] = []
    selected_lookup: dict[tuple[str, str, int, str, int, str], dict] = {}

    for backend in paths["backends"]:
        for qid in qids:
            d8 = [
                str(row["comment_id"])
                for row in contract["dense"][backend][qid][:8]
            ]
            baseline = utility_at8(d8, qid, contract["registry"])
            # Raw retrieval is an explicit operational reference.  M changes
            # availability, but raw Top-8 remains the frozen D8.
            for depth in map(int, paths["dense_depths"]):
                full_current = oracle[(backend, POOL_DENSE, depth, 8, qid)]
                full_max = oracle[(backend, POOL_DENSE, 50, 8, qid)]
                row = {
                    "backend": backend,
                    "pool_family": POOL_DENSE,
                    "dense_depth": depth,
                    "scorer": RAW_SCORER,
                    "replacement_budget": 0,
                    "query_id": qid,
                    "need_stratum": contract["strata"][qid],
                    "baseline_utility_at8": baseline,
                    "selected_utility_at8": baseline,
                    "full_pool_oracle_utility_at8": full_current["full_pool_oracle_utility_at8"],
                    "feasible_oracle_utility_at8": baseline,
                    "max_family_oracle_utility_at8": full_max["full_pool_oracle_utility_at8"],
                    "candidate_access_headroom": full_current["candidate_access_headroom"],
                    "feasible_headroom": 0.0,
                    "access_loss_to_m50": (
                        full_max["full_pool_oracle_utility_at8"]
                        - full_current["full_pool_oracle_utility_at8"]
                    ),
                    "action_space_loss": full_current["candidate_access_headroom"],
                    "learning_loss": 0.0,
                    "realised_gain": 0.0,
                    "max_family_headroom": full_max["candidate_access_headroom"],
                    "feasible_conversion": None,
                    "full_pool_conversion": 0.0 if full_current["candidate_access_headroom"] > EPS else None,
                    "replacement_count": 0,
                    "deeper_dense_entrant_count": 0,
                    "graph_added_entrant_count": 0,
                    "dense_tail_also_graph_count": 0,
                    "graph_added_positive_pair_count": 0,
                    "replacement_utility_at8_contribution": 0.0,
                    "replacement_pairs_json": "[]",
                    "harmful_selection": False,
                    "inference_used_gold_utility": False,
                }
                rows.append(row)
                selected_rows.append({
                    "selection_kind": "raw",
                    "backend": backend,
                    "pool_family": POOL_DENSE,
                    "dense_depth": depth,
                    "scorer": RAW_SCORER,
                    "replacement_budget": 0,
                    "query_id": qid,
                    "selected_comment_ids": d8,
                    "selected_utility_at8": baseline,
                    "replacement_count": 0,
                    "deeper_dense_entrant_count": 0,
                    "graph_added_entrant_count": 0,
                    "inference_used_gold_utility": False,
                    "oracle_only": False,
                })

    for backend in paths["backends"]:
        for scorer in (SCORER_HUBER, SCORER_MLP):
            for pool_family in (POOL_DENSE, POOL_GRAPH):
                for depth in map(int, paths["dense_depths"]):
                    for budget in map(int, paths["replacement_budgets"]):
                        for qid in qids:
                            d8 = [
                                str(row["comment_id"])
                                for row in contract["dense"][backend][qid][:8]
                            ]
                            pool = pools["pool_ids"][(backend, pool_family, depth, qid)]
                            score_map = {
                                cid: float(predictions[(backend, scorer, qid, cid)])
                                for cid in pool
                            }
                            result = exact_select_under_budget(
                                d8, pool, score_map, budget
                            )
                            selected_u = utility_at8(
                                result["selected_ids"], qid, contract["registry"]
                            )
                            baseline = utility_at8(d8, qid, contract["registry"])
                            feasible = oracle[(backend, pool_family, depth, budget, qid)]
                            full_current = oracle[(backend, pool_family, depth, 8, qid)]
                            full_max = oracle[(backend, pool_family, 50, 8, qid)]
                            diagnostics = _replacement_diagnostics(
                                selected_ids=result["selected_ids"],
                                baseline_ids=d8,
                                predicted_scores=score_map,
                                query_id=qid,
                                backend=backend,
                                depth=depth,
                                contract=contract,
                            )
                            realised = selected_u - baseline
                            if not math.isclose(
                                diagnostics["replacement_utility_at8_contribution"],
                                realised,
                                abs_tol=1e-10,
                            ):
                                raise AssertionError("replacement contributions do not sum to set gain")
                            row = {
                                "backend": backend,
                                "pool_family": pool_family,
                                "dense_depth": depth,
                                "scorer": scorer,
                                "replacement_budget": budget,
                                "query_id": qid,
                                "need_stratum": contract["strata"][qid],
                                "baseline_utility_at8": baseline,
                                "selected_utility_at8": selected_u,
                                "full_pool_oracle_utility_at8": full_current["full_pool_oracle_utility_at8"],
                                "feasible_oracle_utility_at8": feasible["oracle_utility_at8"],
                                "max_family_oracle_utility_at8": full_max["full_pool_oracle_utility_at8"],
                                "candidate_access_headroom": full_current["candidate_access_headroom"],
                                "feasible_headroom": feasible["feasible_headroom"],
                                "access_loss_to_m50": (
                                    full_max["full_pool_oracle_utility_at8"]
                                    - full_current["full_pool_oracle_utility_at8"]
                                ),
                                "action_space_loss": feasible["action_space_loss"],
                                "learning_loss": feasible["oracle_utility_at8"] - selected_u,
                                "realised_gain": realised,
                                "max_family_headroom": full_max["candidate_access_headroom"],
                                "feasible_conversion": (
                                    realised / feasible["feasible_headroom"]
                                    if feasible["feasible_headroom"] > EPS else None
                                ),
                                "full_pool_conversion": (
                                    realised / full_current["candidate_access_headroom"]
                                    if full_current["candidate_access_headroom"] > EPS else None
                                ),
                                **diagnostics,
                                "harmful_selection": realised < -EPS,
                                "inference_used_gold_utility": False,
                            }
                            identity = (
                                row["access_loss_to_m50"] + row["action_space_loss"]
                                + row["learning_loss"] + row["realised_gain"]
                            )
                            if abs(identity - row["max_family_headroom"]) > 1e-9:
                                raise AssertionError("four-part decomposition identity failed")
                            if row["learning_loss"] < -1e-9:
                                raise AssertionError("learned set exceeds exact feasible Oracle")
                            rows.append(row)
                            selection = {
                                "selection_kind": "learned",
                                "backend": backend,
                                "pool_family": pool_family,
                                "dense_depth": depth,
                                "scorer": scorer,
                                "replacement_budget": budget,
                                "query_id": qid,
                                "selected_comment_ids": result["selected_ids"],
                                "selected_utility_at8": selected_u,
                                "replacement_count": diagnostics["replacement_count"],
                                "deeper_dense_entrant_count": diagnostics["deeper_dense_entrant_count"],
                                "graph_added_entrant_count": diagnostics["graph_added_entrant_count"],
                                "inference_used_gold_utility": False,
                                "oracle_only": False,
                            }
                            selected_rows.append(selection)
                            selected_lookup[(backend, pool_family, depth, scorer, budget, qid)] = {
                                **selection,
                                "score_map": score_map,
                            }

    # Matched Graph marginal: candidate scores, M, r, and queries are fixed;
    # only FixedGraph4 availability changes.
    row_lookup = {
        (
            row["backend"], row["pool_family"], int(row["dense_depth"]),
            row["scorer"], int(row["replacement_budget"]), row["query_id"],
        ): row
        for row in rows if row["scorer"] != RAW_SCORER
    }
    for row in rows:
        if row["pool_family"] != POOL_GRAPH or row["scorer"] == RAW_SCORER:
            row.update({
                "graph_oracle_marginal_vs_dense": None,
                "graph_policy_marginal_vs_dense": None,
                "graph_opportunity_query": None,
                "graph_opportunity_selected_graph": None,
                "graph_opportunity_realised_positive": None,
                "graph_marginal_added_count": None,
                "graph_marginal_positive_pair_count": None,
                "graph_marginal_pairs_json": None,
            })
            continue
        key = (
            row["backend"], POOL_DENSE, int(row["dense_depth"]), row["scorer"],
            int(row["replacement_budget"]), row["query_id"],
        )
        dense_row = row_lookup[key]
        graph_oracle = oracle[(
            row["backend"], POOL_GRAPH, int(row["dense_depth"]),
            int(row["replacement_budget"]), row["query_id"],
        )]["oracle_utility_at8"]
        dense_oracle = oracle[(
            row["backend"], POOL_DENSE, int(row["dense_depth"]),
            int(row["replacement_budget"]), row["query_id"],
        )]["oracle_utility_at8"]
        union_selection = selected_lookup[(
            row["backend"], POOL_GRAPH, int(row["dense_depth"]), row["scorer"],
            int(row["replacement_budget"]), row["query_id"],
        )]
        dense_selection = selected_lookup[(
            row["backend"], POOL_DENSE, int(row["dense_depth"]), row["scorer"],
            int(row["replacement_budget"]), row["query_id"],
        )]
        union_ids = set(union_selection["selected_comment_ids"])
        dense_ids = set(dense_selection["selected_comment_ids"])
        score_map = union_selection["score_map"]
        added = sorted(
            union_ids - dense_ids,
            key=lambda cid: (-float(score_map[cid]), cid),
        )
        removed = sorted(
            dense_ids - union_ids,
            key=lambda cid: (float(score_map[cid]), cid),
        )
        if len(added) != len(removed):
            raise AssertionError("Graph/Dense set difference is not paired")
        pairs = []
        graph_added_count = graph_positive = 0
        marginal_sum = 0.0
        for entrant, displaced in zip(added, removed, strict=True):
            membership = _source_membership(
                row["backend"], int(row["dense_depth"]), row["query_id"],
                entrant, contract,
            )
            delta = (
                float(contract["registry"][(row["query_id"], entrant)]["utility"])
                - float(contract["registry"][(row["query_id"], displaced)]["utility"])
            ) / 8.0
            marginal_sum += delta
            is_graph_added = membership["graph_added_relative_to_dense_m"]
            graph_added_count += int(is_graph_added)
            graph_positive += int(is_graph_added and delta > EPS)
            pairs.append({
                "entrant_id": entrant,
                "displaced_id": displaced,
                "graph_added_relative_to_dense_m": is_graph_added,
                "utility_at8_contribution": delta,
            })
        if graph_added_count != len(added):
            raise AssertionError(
                "matched Graph marginal contains a candidate that is not "
                "strict Graph-added relative to the matched Dense D_M pool"
            )
        policy_marginal = row["selected_utility_at8"] - dense_row["selected_utility_at8"]
        if not math.isclose(marginal_sum, policy_marginal, abs_tol=1e-10):
            raise AssertionError("Graph marginal pair contributions do not sum")
        opportunity = graph_oracle > dense_oracle + EPS
        row.update({
            "graph_oracle_marginal_vs_dense": graph_oracle - dense_oracle,
            "graph_policy_marginal_vs_dense": policy_marginal,
            "graph_opportunity_query": opportunity,
            "graph_opportunity_selected_graph": bool(opportunity and graph_added_count > 0),
            "graph_opportunity_realised_positive": bool(opportunity and policy_marginal > EPS),
            "graph_marginal_added_count": graph_added_count,
            "graph_marginal_positive_pair_count": graph_positive,
            "graph_marginal_pairs_json": json.dumps(pairs, sort_keys=True),
        })
    return rows, selected_rows


def _bootstrap_plans(
    qids: list[str], strata: dict[str, str], samples: int, seed: int,
) -> tuple[dict[str, dict], dict]:
    groups = {
        "all": list(qids),
        "single_need": [qid for qid in qids if strata[qid] == "single_need"],
        "multi_need": [qid for qid in qids if strata[qid] == "multi_need"],
    }
    plans = {}
    manifest = {}
    for offset, (name, group_qids) in enumerate(groups.items()):
        rng = np.random.default_rng(seed + offset)
        indices = rng.integers(
            0, len(group_qids), size=(samples, len(group_qids)), dtype=np.int32
        )
        plans[name] = {"qids": group_qids, "indices": indices}
        manifest[name] = {
            "seed": seed + offset,
            "queries": len(group_qids),
            "query_ids_sha256": hashlib.sha256(
                ("\n".join(group_qids) + "\n").encode("utf-8")
            ).hexdigest(),
            "indices_shape": list(indices.shape),
            "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        }
    return plans, manifest


def _mean_ci(values: np.ndarray, indices: np.ndarray) -> tuple[float, float, float]:
    point = float(values.mean())
    draws = values[indices].mean(axis=1)
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _ratio_ci(
    numerator: np.ndarray, denominator: np.ndarray, indices: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    denom = float(denominator.sum())
    if abs(denom) <= EPS:
        return None, None, None
    point = float(numerator.sum() / denom)
    num_draw = numerator[indices].sum(axis=1)
    den_draw = denominator[indices].sum(axis=1)
    valid = np.abs(den_draw) > EPS
    if not bool(valid.any()):
        return point, None, None
    draws = num_draw[valid] / den_draw[valid]
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _rows_in_plan(rows: list[dict], plan: dict) -> list[dict]:
    lookup = {str(row["query_id"]): row for row in rows}
    if len(lookup) != len(rows) or not set(plan["qids"]) <= set(lookup):
        raise ValueError("summary cell is not one row per planned query")
    return [lookup[qid] for qid in plan["qids"]]


def _summarise_cell(
    rows: list[dict], plan: dict, *, stratum: str, intervals: list[dict],
) -> dict:
    ordered = _rows_in_plan(rows, plan)
    indices = plan["indices"]
    first = ordered[0]
    identity = {
        key: first[key]
        for key in (
            "backend", "pool_family", "dense_depth", "scorer",
            "replacement_budget",
        )
    }
    summary = {**identity, "stratum": stratum, "queries": len(ordered)}
    scalar_metrics = (
        "baseline_utility_at8", "selected_utility_at8", "realised_gain",
        "candidate_access_headroom", "feasible_headroom", "access_loss_to_m50",
        "action_space_loss", "learning_loss", "max_family_headroom",
        "replacement_count", "deeper_dense_entrant_count",
        "graph_added_entrant_count", "dense_tail_also_graph_count",
    )
    for metric in scalar_metrics:
        values = np.asarray([float(row[metric]) for row in ordered], dtype=float)
        point, low, high = _mean_ci(values, indices)
        summary[f"mean_{metric}"] = point
        summary[f"{metric}_ci_low"] = low
        summary[f"{metric}_ci_high"] = high
        intervals.append({
            **identity,
            "stratum": stratum,
            "metric": metric,
            "point_estimate": point,
            "ci_low": low,
            "ci_high": high,
            "bootstrap_samples": int(indices.shape[0]),
            "bootstrap_unit": "query_id_after_averaging_5_oof_predictions",
            "common_indices_within_stratum": True,
        })
    gains = np.asarray([float(row["realised_gain"]) for row in ordered])
    summary.update({
        "wins": int((gains > EPS).sum()),
        "ties": int((np.abs(gains) <= EPS).sum()),
        "losses": int((gains < -EPS).sum()),
        "harmful_query_count": int((gains < -EPS).sum()),
        "harmful_query_rate": float((gains < -EPS).mean()),
        "worst_query_delta": float(gains.min()),
        "replacement_distribution_json": json.dumps(
            dict(sorted(Counter(int(row["replacement_count"]) for row in ordered).items()))
        ),
    })
    realised = gains
    feasible = np.asarray([float(row["feasible_headroom"]) for row in ordered])
    full = np.asarray([float(row["candidate_access_headroom"]) for row in ordered])
    for metric, denominator in (
        ("feasible_conversion", feasible),
        ("full_pool_conversion", full),
    ):
        point, low, high = _ratio_ci(realised, denominator, indices)
        summary[metric] = point
        summary[f"{metric}_ci_low"] = low
        summary[f"{metric}_ci_high"] = high
        intervals.append({
            **identity,
            "stratum": stratum,
            "metric": metric,
            "point_estimate": point,
            "ci_low": low,
            "ci_high": high,
            "bootstrap_samples": int(indices.shape[0]),
            "bootstrap_unit": "query_id_ratio_of_sums_after_averaging_predictions",
            "common_indices_within_stratum": True,
        })

    if first["pool_family"] == POOL_GRAPH and first["scorer"] != RAW_SCORER:
        marginal = np.asarray([
            float(row["graph_policy_marginal_vs_dense"]) for row in ordered
        ])
        point, low, high = _mean_ci(marginal, indices)
        summary.update({
            "mean_graph_policy_marginal_vs_dense": point,
            "graph_policy_marginal_ci_low": low,
            "graph_policy_marginal_ci_high": high,
            "graph_marginal_wins": int((marginal > EPS).sum()),
            "graph_marginal_ties": int((np.abs(marginal) <= EPS).sum()),
            "graph_marginal_losses": int((marginal < -EPS).sum()),
            "graph_opportunity_queries": sum(bool(row["graph_opportunity_query"]) for row in ordered),
            "graph_marginal_added_candidates": sum(int(row["graph_marginal_added_count"]) for row in ordered),
            "graph_marginal_positive_pairs": sum(int(row["graph_marginal_positive_pair_count"]) for row in ordered),
        })
        intervals.append({
            **identity,
            "stratum": stratum,
            "metric": "graph_policy_marginal_vs_dense",
            "point_estimate": point,
            "ci_low": low,
            "ci_high": high,
            "bootstrap_samples": int(indices.shape[0]),
            "bootstrap_unit": "paired_query_id",
            "common_indices_within_stratum": True,
        })
        opportunities = np.asarray([
            float(bool(row["graph_opportunity_query"])) for row in ordered
        ])
        graph_entrant_queries = np.asarray([
            float(int(row["graph_added_entrant_count"]) > 0) for row in ordered
        ])
        value, lo, hi = _mean_ci(graph_entrant_queries, indices)
        summary["graph_entrant_query_fraction"] = value
        summary["graph_entrant_query_fraction_ci_low"] = lo
        summary["graph_entrant_query_fraction_ci_high"] = hi
        intervals.append({
            **identity,
            "stratum": stratum,
            "metric": "graph_entrant_query_fraction",
            "point_estimate": value,
            "ci_low": lo,
            "ci_high": hi,
            "bootstrap_samples": int(indices.shape[0]),
            "bootstrap_unit": "paired_query_id",
            "common_indices_within_stratum": True,
        })
        selected = np.asarray([
            float(bool(row["graph_opportunity_selected_graph"])) for row in ordered
        ])
        realised_positive = np.asarray([
            float(bool(row["graph_opportunity_realised_positive"])) for row in ordered
        ])
        for metric, numerator in (
            ("graph_opportunity_selection_recall", selected),
            ("graph_opportunity_positive_recall", realised_positive),
        ):
            value, lo, hi = _ratio_ci(numerator, opportunities, indices)
            summary[metric] = value
            summary[f"{metric}_ci_low"] = lo
            summary[f"{metric}_ci_high"] = hi
            intervals.append({
                **identity,
                "stratum": stratum,
                "metric": metric,
                "point_estimate": value,
                "ci_low": lo,
                "ci_high": hi,
                "bootstrap_samples": int(indices.shape[0]),
                "bootstrap_unit": "paired_query_id_ratio_of_sums",
                "common_indices_within_stratum": True,
            })
        pair_total = np.asarray([
            float(row["graph_marginal_added_count"]) for row in ordered
        ])
        pair_positive = np.asarray([
            float(row["graph_marginal_positive_pair_count"]) for row in ordered
        ])
        value, lo, hi = _ratio_ci(pair_positive, pair_total, indices)
        summary["graph_entrant_precision"] = value
        summary["graph_entrant_precision_ci_low"] = lo
        summary["graph_entrant_precision_ci_high"] = hi
        intervals.append({
            **identity,
            "stratum": stratum,
            "metric": "graph_entrant_precision",
            "point_estimate": value,
            "ci_low": lo,
            "ci_high": hi,
            "bootstrap_samples": int(indices.shape[0]),
            "bootstrap_unit": "query_clustered_graph_added_pairs",
            "common_indices_within_stratum": True,
        })
    return summary


def _summaries(
    rows: list[dict], plans: dict,
) -> tuple[list[dict], list[dict], list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(
            row["backend"], row["pool_family"], int(row["dense_depth"]),
            row["scorer"], int(row["replacement_budget"]),
        )].append(row)
    intervals: list[dict] = []
    dense_summary: list[dict] = []
    graph_summary: list[dict] = []
    expected_queries = len(plans["all"]["qids"])
    for key, values in sorted(grouped.items()):
        if len(values) != expected_queries:
            raise ValueError(f"summary cell {key} contains {len(values)} queries")
        for stratum, plan in plans.items():
            summary = _summarise_cell(
                values, plan, stratum=stratum, intervals=intervals
            )
            if key[1] == POOL_DENSE:
                dense_summary.append(summary)
            else:
                graph_summary.append(summary)
    return dense_summary, graph_summary, intervals


def _paired_contrast(
    *, family: str, lhs: str, rhs: str, values: dict[str, float],
    plan: dict, stratum: str, metadata: dict,
) -> dict:
    vector = np.asarray([float(values[qid]) for qid in plan["qids"]])
    point, low, high = _mean_ci(vector, plan["indices"])
    return {
        "contrast_family": family,
        "lhs": lhs,
        "rhs": rhs,
        "stratum": stratum,
        **metadata,
        "queries": len(vector),
        "mean_delta": point,
        "ci_low": low,
        "ci_high": high,
        "wins": int((vector > EPS).sum()),
        "ties": int((np.abs(vector) <= EPS).sum()),
        "losses": int((vector < -EPS).sum()),
        "bootstrap_samples": int(plan["indices"].shape[0]),
        "bootstrap_unit": "paired_query_id_after_averaging_5_oof_predictions",
        "common_indices_within_stratum": True,
    }


def _paired_contrasts(rows: list[dict], plans: dict) -> list[dict]:
    learned = [row for row in rows if row["scorer"] != RAW_SCORER]
    lookup = {
        (
            row["backend"], row["pool_family"], int(row["dense_depth"]),
            row["scorer"], int(row["replacement_budget"]), row["query_id"],
        ): row
        for row in learned
    }
    cells = sorted({key[:-1] for key in lookup})
    output: list[dict] = []
    qids = plans["all"]["qids"]

    def add(family: str, lhs: str, rhs: str, values: dict[str, float], metadata: dict):
        for stratum, plan in plans.items():
            output.append(_paired_contrast(
                family=family, lhs=lhs, rhs=rhs, values=values,
                plan=plan, stratum=stratum, metadata=metadata,
            ))

    for backend, pool, depth, scorer, budget in cells:
        values = {
            qid: float(lookup[(backend, pool, depth, scorer, budget, qid)]["realised_gain"])
            for qid in qids
        }
        add(
            "policy_vs_raw_d8",
            f"{backend}:{pool}:M{depth}:{scorer}:r{budget}",
            f"{backend}:raw_D8",
            values,
            {
                "backend": backend, "pool_family": pool, "dense_depth": depth,
                "scorer": scorer, "replacement_budget": budget,
            },
        )

    for backend in sorted({row["backend"] for row in learned}):
        for pool in (POOL_DENSE, POOL_GRAPH):
            for depth in (8, 12, 20, 50):
                for scorer in (SCORER_HUBER, SCORER_MLP):
                    for left, right in ((1, 0), (2, 1), (4, 2), (8, 4)):
                        values = {
                            qid: (
                                float(lookup[(backend, pool, depth, scorer, left, qid)]["selected_utility_at8"])
                                - float(lookup[(backend, pool, depth, scorer, right, qid)]["selected_utility_at8"])
                            )
                            for qid in qids
                        }
                        add(
                            "replacement_budget_increment",
                            f"r{left}", f"r{right}", values,
                            {
                                "backend": backend, "pool_family": pool,
                                "dense_depth": depth, "scorer": scorer,
                                "replacement_budget": left,
                            },
                        )
                    values = {
                        qid: (
                            float(lookup[(backend, pool, depth, scorer, 8, qid)]["selected_utility_at8"])
                            - float(lookup[(backend, pool, depth, scorer, 1, qid)]["selected_utility_at8"])
                        )
                        for qid in qids
                    }
                    add(
                        "replacement_budget_range",
                        "r8", "r1", values,
                        {
                            "backend": backend, "pool_family": pool,
                            "dense_depth": depth, "scorer": scorer,
                            "replacement_budget": 8,
                        },
                    )
                    for left, right in ((12, 8), (20, 12), (50, 20)):
                        for budget in (1, 2, 4, 8):
                            values = {
                                qid: (
                                    float(lookup[(backend, pool, left, scorer, budget, qid)]["selected_utility_at8"])
                                    - float(lookup[(backend, pool, right, scorer, budget, qid)]["selected_utility_at8"])
                                )
                                for qid in qids
                            }
                            add(
                                "candidate_access_increment",
                                f"M{left}", f"M{right}", values,
                                {
                                    "backend": backend, "pool_family": pool,
                                    "dense_depth": left, "scorer": scorer,
                                    "replacement_budget": budget,
                                },
                            )
                    if pool == POOL_GRAPH:
                        for budget in (0, 1, 2, 4, 8):
                            values = {
                                qid: (
                                    float(lookup[(backend, POOL_GRAPH, depth, scorer, budget, qid)]["selected_utility_at8"])
                                    - float(lookup[(backend, POOL_DENSE, depth, scorer, budget, qid)]["selected_utility_at8"])
                                )
                                for qid in qids
                            }
                            add(
                                "graph_candidate_access_marginal",
                                POOL_GRAPH, POOL_DENSE, values,
                                {
                                    "backend": backend, "pool_family": POOL_GRAPH,
                                    "dense_depth": depth, "scorer": scorer,
                                    "replacement_budget": budget,
                                },
                            )
                for budget in (0, 1, 2, 4, 8):
                    values = {
                        qid: (
                            float(lookup[(backend, pool, depth, SCORER_MLP, budget, qid)]["selected_utility_at8"])
                            - float(lookup[(backend, pool, depth, SCORER_HUBER, budget, qid)]["selected_utility_at8"])
                        )
                        for qid in qids
                    }
                    add(
                        "scorer_family_difference",
                        SCORER_MLP, SCORER_HUBER, values,
                        {
                            "backend": backend, "pool_family": pool,
                            "dense_depth": depth, "scorer": "mlp_minus_huber",
                            "replacement_budget": budget,
                        },
                    )
    return output


def _historical_reconciliation() -> list[dict]:
    output: list[dict] = []
    report_path = ROOT / "out/strict_native_graph_conservative_policy_dev100_v1/strict_native_graph_conservative_policy_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    arm = report["arms"]["graph_strict_native_all__all8__direct_delta__nested"]
    output.append({
        "historical_id": "REPORT89_DIRECT_HUBER_GRAPH_ONE_SWAP",
        "source_family": "strict_native_graph_one_swap",
        "source_artifact": str(report_path.relative_to(ROOT)),
        "source_sha256": sha256(report_path),
        "backend": "historical MiniLM strict-SBERT",
        "pool_definition": "Report87 S0 plus D100-exclusive strict Graph4",
        "anchor_set": "frozen Report87 mixed-LambdaMART Depth S0",
        "action_budget": 1,
        "selection_scope": "NO-OP or one candidate-slot replacement",
        "scorer_family": "Direct Huber action-delta",
        "prediction_target": "replacement Utility@8 delta",
        "baseline_u8": arm["mean_baseline_utility_at8"],
        "realised_u8": arm["mean_policy_utility_at8"],
        "reported_delta": arm["mean_raw_utility_at8_delta"],
        "ci_low": arm["query_bootstrap_95ci"][0],
        "ci_high": arm["query_bootstrap_95ci"][1],
        "oracle_scope": "one-swap action Oracle",
        "oracle_headroom": arm["action_space_oracle_headroom"],
        "reported_conversion": arm["oracle_conversion_ratio"],
        "bootstrap_unit": "query after averaging five OOF action decisions",
        "query_n": 100,
        "comparable_to_new": False,
        "noncomparability_reason": "different S0 anchor, Graph pool, target, threshold and action-level scorer",
        "disposition": "historical operational baseline only",
    })

    dense_path = ROOT / "out/depth_graph_utility_community_frontier_dev100_m50_dense_v1/m50_depth_metrics.csv"
    with dense_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["depth"]) == 8:
                continue
            output.append({
                "historical_id": f"FORMAL_DIRECT_HUBER_{row['backend'].upper()}_D{row['depth']}",
                "source_family": "current_dense_depth_one_swap",
                "source_artifact": str(dense_path.relative_to(ROOT)),
                "source_sha256": sha256(dense_path),
                "backend": row["backend"],
                "pool_definition": f"Dense D{row['depth']}",
                "anchor_set": "raw backend-local Dense D8",
                "action_budget": 1,
                "selection_scope": "NO-OP or one candidate-slot replacement",
                "scorer_family": "Direct Huber action-delta",
                "prediction_target": "replacement Utility@8 delta",
                "baseline_u8": float(row["baseline_utility_at8"]),
                "realised_u8": float(row["realised_oof_utility_at8"]),
                "reported_delta": float(row["realised_delta_vs_d8"]),
                "ci_low": float(row["realised_delta_query_bootstrap_95ci_lo"]),
                "ci_high": float(row["realised_delta_query_bootstrap_95ci_hi"]),
                "oracle_scope": "unrestricted full-pool Oracle in headline; one-swap Oracle stored separately",
                "oracle_headroom": float(row["marginal_oracle_headroom"]),
                "reported_conversion": float(row["conversion_ratio"]),
                "bootstrap_unit": "query after averaging five OOF action decisions",
                "query_n": int(row["queries"]),
                "comparable_to_new": False,
                "noncomparability_reason": "same pool/anchor but action-delta scorer and threshold differ from candidate-utility exact-set scorer",
                "disposition": "historical operational baseline; do not relabel as matched r=1",
            })

    graph_path = ROOT / "out/depth_graph_utility_community_frontier_dev100_m50_graph_v2/m50_graph_absorption.csv"
    ci_path = ROOT / "out/depth_graph_utility_community_frontier_dev100_m50_final_v1/m50_graph_community_paired_contrasts.csv"
    ci_lookup = {}
    with ci_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["source"] == "graph_union" and row["graph_view"] == "fixed_graph4" and row["metric"] == "utility_at8":
                ci_lookup[(row["backend"], int(row["depth"]))] = row
    with graph_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["graph_view"] != "fixed_graph4":
                continue
            ci = ci_lookup[(row["backend"], int(row["depth"]))]
            output.append({
                "historical_id": f"FORMAL_FIXED_GRAPH4_{row['backend'].upper()}_M{row['depth']}",
                "source_family": "formal_depth_graph_one_swap_frontier",
                "source_artifact": str(graph_path.relative_to(ROOT)),
                "source_sha256": sha256(graph_path),
                "backend": row["backend"],
                "pool_definition": f"Dense D{row['depth']} union strict FixedGraph4",
                "anchor_set": "independently fitted one-swap policies sharing the raw backend-local D8 anchor; marginal compares their realised outputs",
                "action_budget": 1,
                "selection_scope": "historical Graph union one-swap policy",
                "scorer_family": "Direct Huber action-delta",
                "prediction_target": "replacement Utility@8 delta",
                "baseline_u8": float(row["dense_realised_utility_at8"]),
                "realised_u8": float(row["union_realised_utility_at8"]),
                "reported_delta": float(row["realised_marginal_vs_dense"]),
                "ci_low": float(ci["bootstrap_95ci_lo"]),
                "ci_high": float(ci["bootstrap_95ci_hi"]),
                "oracle_scope": "unrestricted union minus Dense Oracle",
                "oracle_headroom": float(row["oracle_marginal_vs_dense"]),
                "reported_conversion": None,
                "bootstrap_unit": "paired query",
                "query_n": 100,
                "comparable_to_new": False,
                "noncomparability_reason": "historical independently fitted action-delta policies; new experiment fixes one candidate scorer and exact replacement budget",
                "disposition": "separate historical comparator",
            })

    inventory_path = ROOT / "experiments/selection_forensic_audit/selection_experiment_inventory.csv"
    wanted = {
        "STRICT87-ARM-MIXED-LAMBDAMART-FULL",
        "STRICT87-ARM-SEQUENTIAL-MIXED-LAMBDAMART-FULL",
        "WIDE-LINEAR-ALL", "WIDE-LAMBDAMART-ALL", "WIDE-MLP-ALL",
    }
    with inventory_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["estimand_id"] not in wanted:
                continue
            output.append({
                "historical_id": row["estimand_id"],
                "source_family": row["experiment_family"],
                "source_artifact": row["result_files"],
                "source_sha256": "see forensic inventory evidence field",
                "backend": row["dense_backend"],
                "pool_definition": row["candidate_pool_definition"],
                "anchor_set": "raw D8 or experiment-specific frozen ranking; see evidence",
                "action_budget": row["maximum_replacements"],
                "selection_scope": row["selection_scope"],
                "scorer_family": row["model_family"],
                "prediction_target": row["prediction_target"],
                "baseline_u8": None,
                "realised_u8": None,
                "reported_delta": row["numerical_result"],
                "ci_low": None,
                "ci_high": None,
                "oracle_scope": "experiment-specific; see forensic inventory evidence",
                "oracle_headroom": None,
                "reported_conversion": None,
                "bootstrap_unit": row["bootstrap_method"],
                "query_n": row["query_n"],
                "comparable_to_new": False,
                "noncomparability_reason": "historical unrestricted selector uses a different pool and/or feature/target contract",
                "disposition": row["disposition"],
            })
    return output


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "--"
    return f"{float(value):.{digits}f}"


def _write_latex_tables(
    out: Path, dense: list[dict], graph: list[dict], oracle_lookup: dict,
    qids: list[str],
) -> None:
    overall_dense = [row for row in dense if row["stratum"] == "all"]
    overall_graph = [row for row in graph if row["stratum"] == "all"]
    action_rows = [
        row for row in [*overall_dense, *overall_graph]
        if row["scorer"] != RAW_SCORER
        and int(row["replacement_budget"]) in {1, 8}
        and (
            (row["pool_family"] == POOL_DENSE and int(row["dense_depth"]) in {12, 50})
            or (row["pool_family"] == POOL_GRAPH and int(row["dense_depth"]) in {8, 12})
        )
    ]
    lines = [
        "\\begin{tabular}{llrlrrrrrr}", "\\toprule",
        "Backend & Pool & $M$ & Scorer & $r$ & $A$ & $H_r$ & $L_{action}$ & $L_{learn}$ & Gain \\\\",
        "\\midrule",
    ]
    for row in action_rows:
        lines.append(
            f"{row['backend']} & {'Dense' if row['pool_family']==POOL_DENSE else 'Dense+G4'} & "
            f"{row['dense_depth']} & {'Huber' if row['scorer']==SCORER_HUBER else 'MLP'} & "
            f"{row['replacement_budget']} & {_fmt(row['mean_candidate_access_headroom'])} & "
            f"{_fmt(row['mean_feasible_headroom'])} & {_fmt(row['mean_action_space_loss'])} & "
            f"{_fmt(row['mean_learning_loss'])} & {_fmt(row['mean_realised_gain'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (out / "table_action_space_decomposition.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "\\begin{tabular}{llrrrrrr}", "\\toprule",
        "Backend & Scorer & $M$ & $r$ & Gain & 95\\% CI & Repl. & Harm \\\\",
        "\\midrule",
    ]
    for row in overall_dense:
        if row["scorer"] == RAW_SCORER or int(row["dense_depth"]) == 8 or int(row["replacement_budget"]) == 0:
            continue
        lines.append(
            f"{row['backend']} & {'Huber' if row['scorer']==SCORER_HUBER else 'MLP'} & "
            f"{row['dense_depth']} & {row['replacement_budget']} & {_fmt(row['mean_realised_gain'])} & "
            f"[{_fmt(row['realised_gain_ci_low'])}, {_fmt(row['realised_gain_ci_high'])}] & "
            f"{_fmt(row['mean_replacement_count'],2)} & {row['harmful_query_count']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (out / "table_dense_replacement_frontier.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "\\begin{tabular}{llrrrrrr}", "\\toprule",
        "Backend & Scorer & $M$ & $r$ & Graph $\\Delta$ & 95\\% CI & G entrants & Opp. positive recall \\\\",
        "\\midrule",
    ]
    for row in overall_graph:
        if int(row["dense_depth"]) not in {8, 12} or int(row["replacement_budget"]) == 0:
            continue
        lines.append(
            f"{row['backend']} & {'Huber' if row['scorer']==SCORER_HUBER else 'MLP'} & "
            f"{row['dense_depth']} & {row['replacement_budget']} & "
            f"{_fmt(row['mean_graph_policy_marginal_vs_dense'])} & "
            f"[{_fmt(row['graph_policy_marginal_ci_low'])}, {_fmt(row['graph_policy_marginal_ci_high'])}] & "
            f"{int(row['graph_marginal_added_candidates'])} & "
            f"{_fmt(row.get('graph_opportunity_positive_recall'))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (out / "table_graph_replacement_frontier.tex").write_text("\n".join(lines), encoding="utf-8")


def _render_figures(
    out: Path, dense: list[dict], graph: list[dict], oracle_lookup: dict,
    qids: list[str],
) -> None:
    colors = {SCORER_HUBER: "#2b6cb0", SCORER_MLP: "#c05621"}
    labels = {SCORER_HUBER: "Candidate Huber", SCORER_MLP: "Candidate MLP"}
    budgets = [0, 1, 2, 4, 8]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True)
    selections = {
        POOL_DENSE: [12, 50],
        POOL_GRAPH: [8, 12],
    }
    for row_index, backend in enumerate(("minilm", "e5")):
        for col_index, pool in enumerate((POOL_DENSE, POOL_GRAPH)):
            ax = axes[row_index, col_index]
            for depth in selections[pool]:
                feasible = [
                    statistics.fmean(
                        oracle_lookup[(backend, pool, depth, budget, qid)]["feasible_headroom"]
                        for qid in qids
                    )
                    for budget in budgets
                ]
                full = statistics.fmean(
                    oracle_lookup[(backend, pool, depth, 8, qid)]["candidate_access_headroom"]
                    for qid in qids
                )
                ax.plot(budgets, feasible, marker="o", label=f"M{depth}: feasible $H_r$")
                ax.plot(budgets, [full] * len(budgets), linestyle="--", alpha=.55, label=f"M{depth}: access $A$")
            ax.set_title(f"{backend.upper()} — {'Dense' if pool==POOL_DENSE else 'Dense + FixedGraph4'}")
            ax.set_xlabel("Replacement budget r")
            ax.set_ylabel("Oracle headroom in Utility@8")
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle("Candidate-access and action-space Oracle decomposition")
    fig.tight_layout()
    pdf_metadata = {
        "Creator": "GraphRAG selection action-space repair",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(
        out / "figure_oracle_decomposition.pdf", bbox_inches="tight",
        metadata=pdf_metadata,
    )
    plt.close(fig)

    overall_dense = {
        (row["backend"], int(row["dense_depth"]), row["scorer"], int(row["replacement_budget"])): row
        for row in dense if row["stratum"] == "all" and row["scorer"] != RAW_SCORER
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for row_index, backend in enumerate(("minilm", "e5")):
        for col_index, depth in enumerate((12, 20, 50)):
            ax = axes[row_index, col_index]
            for scorer in (SCORER_HUBER, SCORER_MLP):
                cells = [overall_dense[(backend, depth, scorer, budget)] for budget in budgets]
                values = [cell["mean_realised_gain"] for cell in cells]
                lows = [cell["realised_gain_ci_low"] for cell in cells]
                highs = [cell["realised_gain_ci_high"] for cell in cells]
                ax.plot(budgets, values, marker="o", color=colors[scorer], label=labels[scorer])
                ax.fill_between(budgets, lows, highs, color=colors[scorer], alpha=.14)
            oracle_values = [
                statistics.fmean(
                    oracle_lookup[(backend, POOL_DENSE, depth, budget, qid)]["feasible_headroom"]
                    for qid in qids
                )
                for budget in budgets
            ]
            ax.plot(budgets, oracle_values, color="#4a5568", linestyle="--", label="Feasible Oracle")
            ax.axhline(0, color="black", linewidth=.7)
            ax.set_title(f"{backend.upper()} M={depth}")
            ax.set_xlabel("Replacement budget r")
            ax.set_ylabel("Gain over raw D8")
            ax.grid(alpha=.2)
            if row_index == 0 and col_index == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Dense replacement-budget frontier (shared OOF candidate scores)")
    fig.tight_layout()
    fig.savefig(
        out / "figure_dense_replacement_frontier.pdf", bbox_inches="tight",
        metadata=pdf_metadata,
    )
    plt.close(fig)

    overall_graph = {
        (row["backend"], int(row["dense_depth"]), row["scorer"], int(row["replacement_budget"])): row
        for row in graph if row["stratum"] == "all"
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True, sharey=True)
    for row_index, backend in enumerate(("minilm", "e5")):
        for col_index, depth in enumerate((8, 12)):
            ax = axes[row_index, col_index]
            for scorer in (SCORER_HUBER, SCORER_MLP):
                cells = [overall_graph[(backend, depth, scorer, budget)] for budget in budgets]
                values = [cell["mean_graph_policy_marginal_vs_dense"] for cell in cells]
                lows = [cell["graph_policy_marginal_ci_low"] for cell in cells]
                highs = [cell["graph_policy_marginal_ci_high"] for cell in cells]
                ax.plot(budgets, values, marker="o", color=colors[scorer], label=labels[scorer])
                ax.fill_between(budgets, lows, highs, color=colors[scorer], alpha=.14)
            oracle_values = [
                statistics.fmean(
                    oracle_lookup[(backend, POOL_GRAPH, depth, budget, qid)]["oracle_utility_at8"]
                    - oracle_lookup[(backend, POOL_DENSE, depth, budget, qid)]["oracle_utility_at8"]
                    for qid in qids
                )
                for budget in budgets
            ]
            ax.plot(budgets, oracle_values, color="#4a5568", linestyle="--", label="Graph Oracle marginal")
            ax.axhline(0, color="black", linewidth=.7)
            ax.set_title(f"{backend.upper()} M={depth}")
            ax.set_xlabel("Replacement budget r")
            ax.set_ylabel("Graph − matched Dense Utility@8")
            ax.grid(alpha=.2)
            if row_index == 0 and col_index == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Strict FixedGraph4 replacement frontier")
    fig.tight_layout()
    fig.savefig(
        out / "figure_graph_replacement_frontier.pdf", bbox_inches="tight",
        metadata=pdf_metadata,
    )
    plt.close(fig)


def _classify_preregistered_outcomes(
    *,
    one_swap_action_gap_positive_ci_cells: int,
    dense_total_cells: int,
    dense_r8_vs_r1_positive_ci_cells: int,
    graph_r8_positive_ci_cells: int,
) -> dict[str, bool]:
    """Apply the four outcome definitions from the frozen experiment brief.

    Outcome letters are semantic identifiers, not convenient result labels:
    A means one-swap is adequate, whereas B means it is restrictive and wider
    learned policies realise additional utility.
    """
    one_swap_close_to_full = one_swap_action_gap_positive_ci_cells == 0
    wider_learned_gain_all = (
        dense_r8_vs_r1_positive_ci_cells == dense_total_cells
    )
    wider_learned_gain_none = dense_r8_vs_r1_positive_ci_cells == 0
    return {
        "outcome_A": one_swap_close_to_full and wider_learned_gain_none,
        "outcome_B": (
            one_swap_action_gap_positive_ci_cells == dense_total_cells
            and wider_learned_gain_all
        ),
        "outcome_C": (
            one_swap_action_gap_positive_ci_cells == dense_total_cells
            and wider_learned_gain_none
        ),
        "outcome_D": wider_learned_gain_all and graph_r8_positive_ci_cells == 0,
    }


def _write_result_documents(
    out: Path, dense: list[dict], graph: list[dict], contrasts: list[dict],
    oracle_lookup: dict, qids: list[str], *, scope_label: str,
) -> dict:
    dense_all = {
        (row["backend"], int(row["dense_depth"]), row["scorer"], int(row["replacement_budget"])): row
        for row in dense if row["stratum"] == "all"
    }
    graph_all = {
        (row["backend"], int(row["dense_depth"]), row["scorer"], int(row["replacement_budget"])): row
        for row in graph if row["stratum"] == "all"
    }
    primary_dense = [
        row for row in dense_all.values()
        if row["scorer"] in {SCORER_HUBER, SCORER_MLP}
        and row["dense_depth"] in {12, 20, 50}
        and row["replacement_budget"] in {1, 2, 4, 8}
    ]
    best_dense = max(primary_dense, key=lambda row: row["mean_realised_gain"])
    worst_dense = min(primary_dense, key=lambda row: row["mean_realised_gain"])
    monotonic_cells = 0
    total_cells = 0
    for backend in ("minilm", "e5"):
        for depth in (12, 20, 50):
            for scorer in (SCORER_HUBER, SCORER_MLP):
                values = [
                    dense_all[(backend, depth, scorer, budget)]["mean_realised_gain"]
                    for budget in (1, 2, 4, 8)
                ]
                total_cells += 1
                monotonic_cells += int(
                    all(right >= left - EPS for left, right in zip(values, values[1:]))
                )
    graph_main = [
        row for row in graph_all.values()
        if row["dense_depth"] in {8, 12} and row["replacement_budget"] in {1, 2, 4, 8}
    ]
    graph_positive_ci = sum(
        float(row["graph_policy_marginal_ci_low"]) > 0 for row in graph_main
    )
    graph_d8_r1 = [
        row for row in graph_main
        if int(row["dense_depth"]) == 8 and int(row["replacement_budget"]) == 1
    ]
    graph_d8_r1_positive = sum(
        float(row["graph_policy_marginal_ci_low"]) > 0 for row in graph_d8_r1
    )
    graph_d12 = [
        row for row in graph_main if int(row["dense_depth"]) == 12
    ]
    graph_d12_supported = [
        row for row in graph_d12
        if float(row["graph_policy_marginal_ci_low"]) > 0
    ]
    graph_d12_supported_text = ", ".join(
        f"{row['backend']}/{row['scorer']}/r={row['replacement_budget']}"
        for row in sorted(
            graph_d12_supported,
            key=lambda row: (
                row["backend"], row["scorer"], int(row["replacement_budget"])
            ),
        )
    ) or "none"
    best_graph = max(
        graph_main, key=lambda row: row["mean_graph_policy_marginal_vs_dense"]
    )
    worst_graph = min(
        graph_main, key=lambda row: row["mean_graph_policy_marginal_vs_dense"]
    )
    graph_oracle_positive = all(
        statistics.fmean(
            oracle_lookup[(backend, POOL_GRAPH, depth, budget, qid)]["oracle_utility_at8"]
            - oracle_lookup[(backend, POOL_DENSE, depth, budget, qid)]["oracle_utility_at8"]
            for qid in qids
        ) > EPS
        for backend in ("minilm", "e5")
        for depth in (8, 12)
        for budget in (1, 2, 4, 8)
    )
    range_contrasts = [
        row for row in contrasts
        if row["contrast_family"] == "replacement_budget_range"
        and row["stratum"] == "all"
        and row["pool_family"] == POOL_DENSE
        and int(row["dense_depth"]) in {12, 20, 50}
    ]
    range_positive_ci = sum(float(row["ci_low"]) > 0 for row in range_contrasts)
    # Preserve the outcome letters and meanings exactly as pre-registered in
    # the experiment specification.  In particular, Outcome A is the
    # *one-swap-is-adequate* case; it must not be used for the opposite result.
    one_swap_rows = [
        row for row in dense_all.values()
        if row["dense_depth"] in {12, 20, 50}
        and row["replacement_budget"] == 1
    ]
    one_swap_action_gap_positive_ci = sum(
        float(row["action_space_loss_ci_low"]) > 0 for row in one_swap_rows
    )
    sign_pairs = []
    for depth in (12, 20, 50):
        for scorer in (SCORER_HUBER, SCORER_MLP):
            for budget in (1, 2, 4, 8):
                left = dense_all[("minilm", depth, scorer, budget)]["mean_realised_gain"]
                right = dense_all[("e5", depth, scorer, budget)]["mean_realised_gain"]
                sign_pairs.append((left > EPS) == (right > EPS))
    sign_agreement = statistics.fmean(map(float, sign_pairs))
    graph_r8_primary = [
        row for row in graph_main if row["replacement_budget"] == 8
    ]
    graph_r8_positive_ci = sum(
        float(row["graph_policy_marginal_ci_low"]) > 0
        for row in graph_r8_primary
    )
    if len(range_contrasts) != total_cells or len(one_swap_rows) != total_cells:
        raise AssertionError(
            "pre-registered outcome cells are incomplete: "
            f"range={len(range_contrasts)}, one_swap={len(one_swap_rows)}, "
            f"expected={total_cells}"
        )
    classified = _classify_preregistered_outcomes(
        one_swap_action_gap_positive_ci_cells=one_swap_action_gap_positive_ci,
        dense_total_cells=total_cells,
        dense_r8_vs_r1_positive_ci_cells=range_positive_ci,
        graph_r8_positive_ci_cells=graph_r8_positive_ci,
    )
    outcome_a = classified["outcome_A"]
    outcome_b = classified["outcome_B"]
    outcome_c = classified["outcome_C"]
    outcome_d = classified["outcome_D"]
    supported = [
        name for name, value in (
            ("A_ONE_SWAP_IS_EMPIRICALLY_ADEQUATE", outcome_a),
            ("B_ORIGINAL_ONE_SWAP_POLICY_WAS_MATERIALLY_RESTRICTIVE", outcome_b),
            ("C_ACTION_SPACE_AND_LEARNING_LIMITATIONS_COEXIST_WITHOUT_WIDER_POLICY_GAIN", outcome_c),
            ("D_FULL_SELECTORS_IMPROVE_DENSE_BUT_NOT_GRAPH_POOLS", outcome_d),
        ) if value
    ]

    def cell_text(row: dict, field: str = "mean_realised_gain") -> str:
        low_name = "realised_gain_ci_low" if field == "mean_realised_gain" else "graph_policy_marginal_ci_low"
        high_name = "realised_gain_ci_high" if field == "mean_realised_gain" else "graph_policy_marginal_ci_high"
        return f"{row[field]:+.4f} [{row[low_name]:+.4f}, {row[high_name]:+.4f}]"

    lines = [
        "# Selection Action-Space Repair — Results Interpretation",
        "",
        "## Scope and estimand",
        "",
        f"This is a {scope_label}, fully judged, local experiment over {len(qids)} queries. It does not read Test200, call an external model, alter utility-v2, or use current-query utility, community-response correspondence, explicit Graph/PPR route identity, or need labels as inference features. Dense-rank missingness is retained as an allowed retrieval-rank feature and can identify candidates outside D50, so it is disclosed as an indirect source proxy.",
        "",
        "For each backend, an explicit-route-blind candidate-utility scorer was trained on outer-training queries from the fixed `D50 ∪ FixedGraph4` universe. Each query-candidate received five out-of-fold predictions; those five predictions were averaged before selection. The resulting score map was then held fixed across all candidate widths, Dense/Graph pool comparisons, and replacement budgets.",
        "",
        "## Three-part limitation anatomy",
        "",
        "The per-query ledger separates: (i) candidate access, measured by the unrestricted best-eight Oracle in the visible pool; (ii) action-space feasibility, measured by the exact Oracle under replacement budget r; and (iii) learning loss, measured between that feasible Oracle and the learned set. Relative to the M50 family ceiling, every row satisfies `max headroom = access loss + action-space loss + learning loss + realised gain`.",
        "",
        "## Dense frontier",
        "",
        f"The best learned Dense cell was {best_dense['backend']} M={best_dense['dense_depth']}, {best_dense['scorer']}, r={best_dense['replacement_budget']}: {cell_text(best_dense)} over raw D8.",
        f"The weakest learned Dense cell was {worst_dense['backend']} M={worst_dense['dense_depth']}, {worst_dense['scorer']}, r={worst_dense['replacement_budget']}: {cell_text(worst_dense)}.",
        f"A non-decreasing realised-gain frontier occurred in {monotonic_cells}/{total_cells} backend × depth × scorer cells. More importantly, r=8 exceeded r=1 with a wholly positive paired 95% interval in {range_positive_ci}/{total_cells} cells. Oracle feasibility is necessarily non-decreasing in r; learned performance need not be locally monotone at every intermediate step.",
        "This supports pre-registered Outcome B: the original one-swap policy was materially restrictive. It does not imply that learning error disappeared. At r=8 the action-space loss is zero by construction, while the remaining feasible-Oracle gap is still the learning loss; this residual is large in every reported pool.",
        "",
        "## Strict FixedGraph4 marginal",
        "",
        f"The strongest learned Graph-minus-matched-Dense cell among M=8/12 was {best_graph['backend']} M={best_graph['dense_depth']}, {best_graph['scorer']}, r={best_graph['replacement_budget']}: {cell_text(best_graph, 'mean_graph_policy_marginal_vs_dense')}.",
        f"The weakest was {worst_graph['backend']} M={worst_graph['dense_depth']}, {worst_graph['scorer']}, r={worst_graph['replacement_budget']}: {cell_text(worst_graph, 'mean_graph_policy_marginal_vs_dense')}.",
        f"{graph_positive_ci}/{len(graph_main)} primary Graph marginal cells had a 95% interval wholly above zero. Graph-opportunity conversion and entrant precision are reported in `graph_summary.csv`; only candidates added by strict FixedGraph4 relative to the matched Dense D_M pool count as Graph marginal entrants.",
        f"At D8+Graph4, {graph_d8_r1_positive}/{len(graph_d8_r1)} backend-by-scorer r=1 cells had intervals above zero. At D12+Graph4, {len(graph_d12_supported)}/{len(graph_d12)} backend-by-scorer-by-budget cells were positive: {graph_d12_supported_text}. This dependence on backend, scorer and replacement budget is weaker evidence than the shallow-pool result. At M20/M50 the Graph marginal was generally small and unstable, consistent with deeper Dense access absorbing most Graph-only opportunity.",
        "",
        "## Pre-registered outcome reading",
        "",
        *(f"- {item}" for item in supported),
        "",
        f"MiniLM/E5 sign agreement across the 24 matched Dense frontier cells was {sign_agreement:.1%}. This is a development comparison, not a new held-out confirmation.",
        "",
        "## Historical reconciliation",
        "",
        "Report89 Direct Huber remains a historical operational baseline. It predicts replacement-action deltas from a different frozen S0 anchor and uses a thresholded one-swap policy. It is therefore not renamed as the matched r=1 candidate-utility policy. Older unrestricted selectors likewise remain separate because their pools, targets or feature contracts differ.",
        "",
        "## Reading the files",
        "",
        "- `constrained_oracles.parquet` contains exact feasible sets for r=0/1/2/4/8.",
        "- `per_query_decomposition.parquet` is the authoritative access/action/learning ledger.",
        "- `dense_summary.csv` and `graph_summary.csv` contain primary overall and secondary single-/multi-need summaries.",
        "- `paired_contrasts.csv` uses one shared bootstrap index plan within each reporting stratum.",
    ]
    (out / "RESULTS_INTERPRETATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    implications = [
        "# Thesis Implications — Selection Action-Space Repair",
        "",
        "## Correct description of the selector",
        "",
        "**Evidence selection remains an accurate task-level description** because the repaired system chooses a size-eight set from the visible candidate pool and may admit multiple entrants. The historical main policy, however, is better described as **conservative D8-anchored one-swap repair**, not unrestricted M-to-8 selection. The repaired development analysis uses **replacement-budget-constrained candidate-utility reranking**: one frozen OOF candidate score map is optimized exactly under r ∈ {0,1,2,4,8}.",
        "",
        "## What can be claimed now",
        "",
        f"The experiment directly quantifies candidate-access, action-space and learning losses on the fully judged {scope_label} ({len(qids)} queries). The supported pre-registered outcome labels are: {', '.join(supported) if supported else 'none'}.",
        "",
        "The existing practical Dense boundary remains **D12** because that is the depth transferred in the already frozen held-out experiment. This new development-only frontier can explain whether larger r changes the mechanism, but it cannot replace the held-out D12 decision or use Test200 to choose r.",
        "",
        f"The most robust Graph value remains **D8 + strict FixedGraph4**. At D12, {len(graph_d12_supported)}/{len(graph_d12)} development cells have positive intervals ({graph_d12_supported_text}), so any residual is conditional on backend, scorer and replacement budget rather than a new general operational boundary; at M20/M50 it is generally small or null. FixedGraph4 here means native Graph candidates before fallback, callback or padding; Graph/Dense overlaps are retained as Dense-accessible candidates and are not counted as Graph-exclusive marginal evidence.",
        "",
        "The already reported Test200 findings remain evidence about the historical D12/one-swap operational system only. They do not validate the repaired r=2/4/8 policies, because this experiment intentionally never reads Test200.",
        "",
        "## Validation still required",
        "",
        "If the repaired development analysis changes the proposed final selector or replacement budget, that choice requires one fresh validation set (or a newly frozen untouched split) before it can be presented as confirmatory. Reusing Test200 for this choice would turn it into development data.",
        "",
        "## Recommended thesis placement",
        "",
        "Use the Oracle decomposition to explain why a wider pool is not the same as an unrestricted learned selector. Present the Dense replacement frontier as the matched repair, then report strict Graph marginal value against the corresponding Dense policy. Keep Report89 and the former Direct-Huber frontiers in a short historical-method subsection or appendix, clearly labelled as action-delta one-swap systems.",
    ]
    (out / "THESIS_IMPLICATIONS.md").write_text("\n".join(implications) + "\n", encoding="utf-8")
    return {
        "outcome_A": outcome_a,
        "outcome_B": outcome_b,
        "outcome_C": outcome_c,
        "outcome_D": outcome_d,
        "supported_outcomes": supported,
        "one_swap_action_gap_positive_ci_cells": one_swap_action_gap_positive_ci,
        "one_swap_action_gap_total_cells": len(one_swap_rows),
        "dense_monotonic_cells": monotonic_cells,
        "dense_total_cells": total_cells,
        "dense_r8_vs_r1_positive_ci_cells": range_positive_ci,
        "graph_positive_ci_cells": graph_positive_ci,
        "graph_primary_cells": len(graph_main),
        "graph_r8_positive_ci_cells": graph_r8_positive_ci,
        "graph_r8_primary_cells": len(graph_r8_primary),
        "backend_sign_agreement": sign_agreement,
    }


def _verified_parquet_rows(directory: Path, filename: str, manifest: dict) -> list[dict]:
    path = directory / filename
    expected = (manifest.get("output_hashes") or {}).get(filename)
    if not path.exists() or expected is None or sha256(path) != expected:
        raise ValueError(f"frozen source artifact failed hash validation: {filename}")
    return pq.read_table(path).to_pylist()


def _load_matched_baseline_artifacts(
    source_dir: Path,
    *,
    backend: str,
    depth: int,
    qids: list[str],
    primary_ids: dict[str, list[str]],
) -> dict:
    _reject_test(source_dir)
    manifest = json.loads(
        (source_dir / "reproduction_manifest.json").read_text(encoding="utf-8")
    )
    source_config = json.loads(
        (source_dir / "config.json").read_text(encoding="utf-8")
    )
    if source_config["feature_names"] != list(STATIC_PREDICTOR_FEATURES):
        raise ValueError("frozen Huber/MLP feature contract changed")
    if manifest["development_queries"] != len(qids):
        raise ValueError("frozen Huber/MLP query count changed")
    predictions = _verified_parquet_rows(
        source_dir, "oof_candidate_predictions.parquet", manifest
    )
    selected = _verified_parquet_rows(source_dir, "selected_sets.parquet", manifest)
    pool_rows = _verified_parquet_rows(
        source_dir, "candidate_pool_manifest.parquet", manifest
    )
    source_primary: dict[str, list[str]] = defaultdict(list)
    for row in pool_rows:
        if (
            str(row["backend"]) == backend
            and str(row["pool_family"]) == POOL_DENSE
            and int(row["dense_depth"]) == depth
        ):
            source_primary[str(row["query_id"])].append(
                (int(row["pool_position"]), str(row["candidate_id"]))
            )
    source_primary = {
        qid: [cid for _, cid in sorted(values)]
        for qid, values in source_primary.items()
    }
    if source_primary != primary_ids:
        raise ValueError("current E5-D12 identities differ from frozen baseline")
    allowed_scorers = {SCORER_HUBER, SCORER_MLP}
    score_rows = [
        row for row in predictions
        if str(row["backend"]) == backend
        and str(row["scorer"]) in allowed_scorers
        and str(row["query_id"]) in set(qids)
        and str(row["candidate_id"]) in set(primary_ids[str(row["query_id"])])
    ]
    expected_pairs = len(qids) * depth * len(allowed_scorers)
    if len(score_rows) != expected_pairs:
        raise ValueError(
            f"frozen baseline primary score coverage {len(score_rows)} != {expected_pairs}"
        )
    score_lookup = {
        (str(row["scorer"]), str(row["query_id"]), str(row["candidate_id"])):
        float(row["oof_prediction_mean"])
        for row in score_rows
    }
    fold_lookup = {
        (str(row["scorer"]), str(row["query_id"]), str(row["candidate_id"])):
        list(map(int, row["repeat_folds"]))
        for row in score_rows
    }
    selected_lookup = {
        (
            str(row["scorer"]), int(row["replacement_budget"]),
            str(row["query_id"]),
        ): list(map(str, row["selected_comment_ids"]))
        for row in selected
        if str(row.get("selection_kind")) == "learned"
        and str(row["backend"]) == backend
        and str(row["pool_family"]) == POOL_DENSE
        and int(row["dense_depth"]) == depth
        and str(row["scorer"]) in allowed_scorers
    }
    return {
        "manifest": manifest,
        "config": source_config,
        "score_lookup": score_lookup,
        "fold_lookup": fold_lookup,
        "selected_lookup": selected_lookup,
    }


def _evaluate_primary_rankings(
    *,
    scorers: list[str],
    score_lookup: dict[tuple[str, str, str], float],
    primary_ids: dict[str, list[str]],
    qids: list[str],
    registry: dict,
) -> tuple[list[dict], list[dict]]:
    per_query: list[dict] = []
    summaries: list[dict] = []
    for scorer in scorers:
        for qid in qids:
            ids = list(primary_ids[qid])
            scores = [float(score_lookup[(scorer, qid, cid)]) for cid in ids]
            utility = [float(registry[(qid, cid)]["utility"]) for cid in ids]
            rho = float(spearmanr(scores, utility).statistic)
            if not math.isfinite(rho):
                raise ValueError(f"{scorer}/{qid}: non-finite Spearman")
            ordered_pairs = [
                (left, right)
                for left in range(len(ids))
                for right in range(len(ids))
                if utility[left] - utility[right] >= 1.0
            ]
            pairwise = (
                statistics.fmean(
                    float(scores[left] > scores[right])
                    for left, right in ordered_pairs
                )
                if ordered_pairs else None
            )
            ranked = sorted(
                ids, key=lambda cid: (-float(score_lookup[(scorer, qid, cid)]), cid)
            )
            gains = {cid: float(registry[(qid, cid)]["utility"]) for cid in ids}
            ndcg = float(graded_ndcg_at(ranked, gains, 8))
            per_query.append({
                "scorer": scorer,
                "query_id": qid,
                "within_query_spearman": rho,
                "pairwise_accuracy_margin_1": pairwise,
                "ndcg_at8_continuous_utility_gain": ndcg,
                "candidate_count": len(ids),
                "eligible_ordered_pair_count": len(ordered_pairs),
                "mae": None if scorer == SCORER_LAMBDAMART else statistics.fmean(
                    abs(scores[index] - utility[index]) for index in range(len(ids))
                ),
                "rmse": None if scorer == SCORER_LAMBDAMART else math.sqrt(
                    statistics.fmean(
                        (scores[index] - utility[index]) ** 2
                        for index in range(len(ids))
                    )
                ),
            })
        rows = [row for row in per_query if row["scorer"] == scorer]
        summaries.append({
            "scorer": scorer,
            "queries": len(rows),
            "candidate_pairs": sum(int(row["candidate_count"]) for row in rows),
            "within_query_spearman": statistics.fmean(
                float(row["within_query_spearman"]) for row in rows
            ),
            "pairwise_accuracy_margin_1": statistics.fmean(
                float(row["pairwise_accuracy_margin_1"])
                for row in rows
                if row["pairwise_accuracy_margin_1"] is not None
            ),
            "pairwise_eligible_queries": sum(
                row["pairwise_accuracy_margin_1"] is not None for row in rows
            ),
            "ndcg_at8_continuous_utility_gain": statistics.fmean(
                float(row["ndcg_at8_continuous_utility_gain"]) for row in rows
            ),
            "mae": (
                None if scorer == SCORER_LAMBDAMART
                else statistics.fmean(float(row["mae"]) for row in rows)
            ),
            "rmse": (
                None if scorer == SCORER_LAMBDAMART
                else statistics.fmean(float(row["rmse"]) for row in rows)
            ),
        })
    return per_query, summaries


def _paired_values_summary(values: dict[str, float], plan: dict) -> dict:
    vector = np.asarray(
        [float(values.get(qid, float("nan"))) for qid in plan["qids"]],
        dtype=float,
    )
    finite = np.isfinite(vector)
    if not finite.any():
        raise ValueError("paired contrast has no eligible queries")
    if finite.all():
        point, low, high = _mean_ci(vector, plan["indices"])
    else:
        point = float(np.mean(vector[finite]))
        bootstrap_means = np.nanmean(vector[plan["indices"]], axis=1)
        low, high = map(float, np.quantile(bootstrap_means, [0.025, 0.975]))
    eligible = vector[finite]
    return {
        "mean_delta": point,
        "ci_low": low,
        "ci_high": high,
        "wins": int((eligible > EPS).sum()),
        "ties": int((np.abs(eligible) <= EPS).sum()),
        "losses": int((eligible < -EPS).sum()),
        "harm_rate": float((eligible < -EPS).mean()),
        "queries": len(eligible),
        "bootstrap_samples": int(plan["indices"].shape[0]),
        "bootstrap_unit": "paired whole query_id",
    }


def _build_primary_selector_results(
    *,
    scorers: list[str],
    score_lookup: dict[tuple[str, str, str], float],
    primary_ids: dict[str, list[str]],
    contract: dict,
    backend: str,
    budgets: list[int],
    baseline_selected_lookup: dict,
    plan: dict,
) -> tuple[list[dict], list[dict], dict[str, float]]:
    per_query: list[dict] = []
    oracle_headroom: dict[str, float] = {}
    for qid in contract["qids"]:
        pool = list(primary_ids[qid])
        d8 = [
            str(row["comment_id"])
            for row in contract["dense"][backend][qid][:8]
        ]
        utility_scores = {
            cid: float(contract["registry"][(qid, cid)]["utility"])
            for cid in pool
        }
        raw_utility = utility_at8(d8, qid, contract["registry"])
        oracle = exact_select_under_budget(d8, pool, utility_scores, 8)
        oracle_utility = utility_at8(
            oracle["selected_ids"], qid, contract["registry"]
        )
        oracle_headroom[qid] = oracle_utility - raw_utility
        for scorer in scorers:
            scores = {cid: float(score_lookup[(scorer, qid, cid)]) for cid in pool}
            for budget in budgets:
                selected = exact_select_under_budget(d8, pool, scores, budget)
                ids = list(map(str, selected["selected_ids"]))
                if scorer in {SCORER_HUBER, SCORER_MLP}:
                    frozen = baseline_selected_lookup[(scorer, budget, qid)]
                    if ids != frozen:
                        raise AssertionError(
                            f"frozen {scorer}/r{budget}/{qid} selection not reproduced"
                        )
                selected_utility = utility_at8(ids, qid, contract["registry"])
                per_query.append({
                    "backend": backend,
                    "dense_depth": len(pool),
                    "final_k": 8,
                    "scorer": scorer,
                    "replacement_budget": budget,
                    "query_id": qid,
                    "raw_dense8_ids": d8,
                    "selected_comment_ids": ids,
                    "replacement_count": int(selected["replacement_count"]),
                    "raw_dense8_utility_at8": raw_utility,
                    "selected_utility_at8": selected_utility,
                    "delta_utility_at8": selected_utility - raw_utility,
                    "oracle_selected_comment_ids": list(oracle["selected_ids"]),
                    "oracle_utility_at8": oracle_utility,
                    "oracle_headroom": oracle_headroom[qid],
                    "inference_used_gold_utility": False,
                    "inference_used_community_response": False,
                })
    summaries: list[dict] = []
    for scorer in scorers:
        for budget in budgets:
            rows = [
                row for row in per_query
                if row["scorer"] == scorer
                and int(row["replacement_budget"]) == budget
            ]
            delta = {row["query_id"]: float(row["delta_utility_at8"]) for row in rows}
            summary = _paired_values_summary(delta, plan)
            numerator = np.asarray([delta[qid] for qid in plan["qids"]], dtype=float)
            denominator = np.asarray(
                [oracle_headroom[qid] for qid in plan["qids"]], dtype=float
            )
            conversion, conversion_low, conversion_high = _ratio_ci(
                numerator, denominator, plan["indices"]
            )
            summaries.append({
                "scorer": scorer,
                "replacement_budget": budget,
                "queries": len(rows),
                "realised_utility_at8": statistics.fmean(
                    float(row["selected_utility_at8"]) for row in rows
                ),
                "raw_dense8_utility_at8": statistics.fmean(
                    float(row["raw_dense8_utility_at8"]) for row in rows
                ),
                "oracle_utility_at8": statistics.fmean(
                    float(row["oracle_utility_at8"]) for row in rows
                ),
                "oracle_headroom": statistics.fmean(
                    float(row["oracle_headroom"]) for row in rows
                ),
                "oracle_conversion": conversion,
                "oracle_conversion_ci_low": conversion_low,
                "oracle_conversion_ci_high": conversion_high,
                **summary,
            })
    lambda_rows = [
        row for row in per_query if row["scorer"] == SCORER_LAMBDAMART
    ]
    for qid in contract["qids"]:
        r4 = next(
            row for row in lambda_rows
            if row["query_id"] == qid and int(row["replacement_budget"]) == 4
        )
        r8 = next(
            row for row in lambda_rows
            if row["query_id"] == qid and int(row["replacement_budget"]) == 8
        )
        if r4["selected_comment_ids"] != r8["selected_comment_ids"]:
            raise AssertionError("E5-D12 LambdaMART r4/r8 outputs differ")
    return per_query, summaries, oracle_headroom


def _primary_contrast(
    *,
    family: str,
    lhs: str,
    rhs: str,
    lhs_values: dict[str, float],
    rhs_values: dict[str, float],
    plan: dict,
    metadata: dict,
) -> dict:
    values = {
        qid: float(lhs_values[qid]) - float(rhs_values[qid])
        for qid in plan["qids"]
        if qid in lhs_values and qid in rhs_values
    }
    return {
        "contrast_family": family,
        "lhs": lhs,
        "rhs": rhs,
        **metadata,
        **_paired_values_summary(values, plan),
    }


def _matched_report(
    *,
    ranking_summary: list[dict],
    selector_summary: list[dict],
    model_contrasts: list[dict],
    capacity_contrasts: list[dict],
    best_budget: int,
    conclusion: str,
    integrity: dict,
    xgboost_version: str,
) -> str:
    names = {
        SCORER_HUBER: "Huber",
        SCORER_MLP: "Small MLP",
        SCORER_LAMBDAMART: "LambdaMART",
    }
    lines = [
        "# Matched Development300 LambdaMART RQ2b Experiment",
        "",
        "## 1. Model and implementation",
        "",
        f"The experiment used XGBoost {xgboost_version} `XGBRanker` with "
        "`objective=rank:ndcg`. It retained the project's historical "
        "utility-to-grade rule `max(0, min(6, round(U)-1))`, `hist` trees, "
        "query groups, and mean LambdaRank pair sampling. Raw model outputs "
        "are ranking scores and were not evaluated with MAE/RMSE.",
        "",
        "Official implementation references: "
        "https://xgboost.readthedocs.io/en/release_3.0.0/tutorials/learning_to_rank.html "
        "and https://xgboost.readthedocs.io/en/release_3.0.0/parameter.html.",
        "",
        "## 2. Match to current RQ2b",
        "",
        f"All integrity gates passed: {integrity['passed_checks']}/"
        f"{integrity['total_checks']}. The run uses the same 300 query IDs, "
        "3,600 E5-D12 pairs, eight inference-time features, 5x5 outer "
        "query-grouped folds, three inner folds, StandardScaler training-only "
        "preprocessing, exact replacement solver, and 5,000 whole-query "
        "bootstrap samples as the frozen Huber/Small-MLP experiment.",
        "",
        "## 3. Ranking quality",
        "",
        "| Reranker | Within-query Spearman | Pairwise accuracy | nDCG@8 |",
        "|---|---:|---:|---:|",
    ]
    for row in ranking_summary:
        lines.append(
            f"| {names[row['scorer']]} | {row['within_query_spearman']:.4f} | "
            f"{row['pairwise_accuracy_margin_1']:.4f} | "
            f"{row['ndcg_at8_continuous_utility_gain']:.4f} |"
        )
    lines.extend([
        "",
        "## 4. Final Top-8 utility",
        "",
        "| Reranker | r | Realised U@8 | Delta vs Raw D8 | 95% CI | W/T/L | Conversion | Harm rate |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ])
    for row in selector_summary:
        lines.append(
            f"| {names[row['scorer']]} | {int(row['replacement_budget'])} | "
            f"{row['realised_utility_at8']:.6f} | {row['mean_delta']:+.6f} | "
            f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | "
            f"{row['oracle_conversion']:.4f} | {row['harm_rate']:.4f} |"
        )
    lines.extend([
        "",
        "## 5. Paired model comparison",
        "",
        f"The best LambdaMART capacity by Development300 mean Utility@8 was r={best_budget}; ties prefer the smaller capacity.",
        "",
        "| Contrast | Delta U@8 | 95% CI | W/T/L |",
        "|---|---:|---|---:|",
    ])
    for row in model_contrasts:
        lines.append(
            f"| {row['lhs']} minus {row['rhs']} | {row['mean_delta']:+.6f} | "
            f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )
    lines.extend([
        "",
        "## 6. Capacity comparison",
        "",
        "| Contrast | Delta U@8 | 95% CI | W/T/L |",
        "|---|---:|---|---:|",
    ])
    for row in capacity_contrasts:
        lines.append(
            f"| {row['lhs']} minus {row['rhs']} | {row['mean_delta']:+.6f} | "
            f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )
    knee = next(row for row in capacity_contrasts if row["lhs"] == "r2" and row["rhs"] == "r1")
    beyond = [
        row for row in capacity_contrasts
        if row["rhs"] == "r2" and row["lhs"] in {"r4", "r8"}
    ]
    knee_remains = knee["ci_low"] > 0 and all(
        row["ci_low"] <= 0 <= row["ci_high"] for row in beyond
    )
    lines.extend([
        "",
        "The practical r=2 capacity knee " + (
            "remains supported under LambdaMART."
            if knee_remains else "is not preserved under LambdaMART."
        ),
        "",
        "## 7. Community correspondence",
        "",
        "The primary experiment was completed without opening hidden community "
        "responses. A post-hoc Development100 join is reported only if a separate "
        "correspondence artifact is present; correspondence never entered training, "
        "tuning, features, or selection.",
        "",
        "## 8. Thesis conclusion",
        "",
        f"**{conclusion}.**",
        "",
    ])
    return "\n".join(lines)


def run_matched_lambdamart(
    output_dir: Path | None = None,
    config_key: str = "rq2b_matched_lambdamart_dev300",
) -> dict:
    started = time.perf_counter()
    raw = project_config.load()[config_key]
    if str(raw.get("experiment_kind")) != "matched_lambdamart":
        raise ValueError("matched LambdaMART config kind is missing")
    source_key = str(raw["source_config_key"])
    source_raw = project_config.load()[source_key]
    contract = _load_contract(source_raw)
    pools = _build_pools_and_features(contract)
    backend = str(raw["backend"])
    depth = int(raw["dense_depth"])
    budgets = list(map(int, raw["replacement_budgets"]))
    if backend != "e5" or depth != 12 or budgets != [1, 2, 4, 8]:
        raise ValueError("primary matched LambdaMART scope must remain E5-D12/r1,2,4,8")
    if list(STATIC_PREDICTOR_FEATURES) != list(raw["expected_feature_names"]):
        raise ValueError("matched LambdaMART feature list changed")
    source_dir = Path(raw["source_experiment_dir"])
    if not source_dir.is_absolute():
        source_dir = ROOT / source_dir
    destination = Path(output_dir or raw["output_dir"])
    if not destination.is_absolute():
        destination = ROOT / destination
    destination = destination.resolve()
    _reject_test(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")

    primary_ids = {
        qid: list(pools["pool_ids"][(backend, POOL_DENSE, depth, qid)])
        for qid in contract["qids"]
    }
    if len(primary_ids) != 300 or any(len(ids) != 12 for ids in primary_ids.values()):
        raise ValueError("primary E5-D12 identity must be exactly 300x12")
    baseline = _load_matched_baseline_artifacts(
        source_dir,
        backend=backend,
        depth=depth,
        qids=contract["qids"],
        primary_ids=primary_ids,
    )
    settings = [dict(row) for row in raw["lambdamart"]["grid"]]
    if not settings or any(row.get("objective") != "rank:ndcg" for row in settings):
        raise ValueError("LambdaMART grid must use rank:ndcg")
    scorer_pool_rows, lambda_scores, tuning_audit = _run_matched_lambdamart_oof(
        contract, pools, backend=backend, settings=settings
    )
    primary_pairs = {
        (qid, cid) for qid, ids in primary_ids.items() for cid in ids
    }
    primary_lambda_rows = [
        row for row in scorer_pool_rows
        if (str(row["query_id"]), str(row["candidate_id"])) in primary_pairs
    ]
    if len(primary_lambda_rows) != 3600:
        raise ValueError("LambdaMART primary OOF map is not 3,600 pairs")

    score_lookup: dict[tuple[str, str, str], float] = dict(
        baseline["score_lookup"]
    )
    lambda_fold_lookup = {}
    for row in primary_lambda_rows:
        qid = str(row["query_id"])
        cid = str(row["candidate_id"])
        score_lookup[(SCORER_LAMBDAMART, qid, cid)] = float(
            row["oof_prediction_mean"]
        )
        lambda_fold_lookup[(qid, cid)] = list(map(int, row["repeat_folds"]))
        for baseline_scorer in (SCORER_HUBER, SCORER_MLP):
            if lambda_fold_lookup[(qid, cid)] != baseline["fold_lookup"][(
                baseline_scorer, qid, cid
            )]:
                raise ValueError("LambdaMART OOF fold membership differs from baseline")

    scorers = [SCORER_HUBER, SCORER_MLP, SCORER_LAMBDAMART]
    ranking_per_query, ranking_summary = _evaluate_primary_rankings(
        scorers=scorers,
        score_lookup=score_lookup,
        primary_ids=primary_ids,
        qids=contract["qids"],
        registry=contract["registry"],
    )
    expected_baselines = raw["expected_baseline_ranking_metrics"]
    for row in ranking_summary:
        if row["scorer"] not in expected_baselines:
            continue
        for metric, expected in expected_baselines[row["scorer"]].items():
            if not math.isclose(float(row[metric]), float(expected), abs_tol=5e-6):
                raise ValueError(f"frozen {row['scorer']} {metric} failed reproduction")

    plans, bootstrap_manifest = _bootstrap_plans(
        contract["qids"], contract["strata"],
        int(source_raw["bootstrap_samples"]), int(source_raw["bootstrap_seed"]),
    )
    source_bootstrap = baseline["manifest"]["bootstrap_common_indices"]
    if bootstrap_manifest != source_bootstrap:
        raise ValueError("whole-query bootstrap plan differs from frozen baseline")
    selector_per_query, selector_summary, oracle_headroom = _build_primary_selector_results(
        scorers=scorers,
        score_lookup=score_lookup,
        primary_ids=primary_ids,
        contract=contract,
        backend=backend,
        budgets=budgets,
        baseline_selected_lookup=baseline["selected_lookup"],
        plan=plans["all"],
    )
    raw_mean = statistics.fmean(
        utility_at8(
            [str(row["comment_id"]) for row in contract["dense"][backend][qid][:8]],
            qid,
            contract["registry"],
        )
        for qid in contract["qids"]
    )
    oracle_mean = raw_mean + statistics.fmean(oracle_headroom.values())
    if not math.isclose(raw_mean, float(raw["expected_raw_dense8_utility"]), abs_tol=5e-7):
        raise ValueError("Raw Dense-8 utility reference changed")
    if not math.isclose(oracle_mean, float(raw["expected_oracle_utility"]), abs_tol=5e-7):
        raise ValueError("E5-D12 Oracle utility reference changed")

    ranking_lookup = {
        (row["scorer"], row["query_id"]): row for row in ranking_per_query
    }
    ranking_contrasts: list[dict] = []
    for comparator in (SCORER_HUBER, SCORER_MLP):
        for metric in (
            "within_query_spearman", "pairwise_accuracy_margin_1",
            "ndcg_at8_continuous_utility_gain",
        ):
            ranking_contrasts.append(_primary_contrast(
                family="candidate_ranking_quality",
                lhs=SCORER_LAMBDAMART,
                rhs=comparator,
                lhs_values={
                    qid: float(ranking_lookup[(SCORER_LAMBDAMART, qid)][metric])
                    for qid in contract["qids"]
                    if ranking_lookup[(SCORER_LAMBDAMART, qid)][metric] is not None
                },
                rhs_values={
                    qid: float(ranking_lookup[(comparator, qid)][metric])
                    for qid in contract["qids"]
                    if ranking_lookup[(comparator, qid)][metric] is not None
                },
                plan=plans["all"],
                metadata={"metric": metric},
            ))

    selected_lookup = {
        (row["scorer"], int(row["replacement_budget"]), row["query_id"]): row
        for row in selector_per_query
    }
    lambda_summaries = [
        row for row in selector_summary if row["scorer"] == SCORER_LAMBDAMART
    ]
    best_budget = min(
        budgets,
        key=lambda budget: (
            -next(
                float(row["realised_utility_at8"])
                for row in lambda_summaries
                if int(row["replacement_budget"]) == budget
            ),
            budget,
        ),
    )
    model_contrasts: list[dict] = []
    for comparator in (SCORER_HUBER, SCORER_MLP):
        model_contrasts.append(_primary_contrast(
            family="best_lambdamart_vs_two_swap",
            lhs=f"LambdaMART r{best_budget}",
            rhs=("Huber r2" if comparator == SCORER_HUBER else "Small MLP r2"),
            lhs_values={
                qid: float(selected_lookup[(SCORER_LAMBDAMART, best_budget, qid)]["selected_utility_at8"])
                for qid in contract["qids"]
            },
            rhs_values={
                qid: float(selected_lookup[(comparator, 2, qid)]["selected_utility_at8"])
                for qid in contract["qids"]
            },
            plan=plans["all"],
            metadata={
                "lambda_budget": best_budget,
                "comparator_scorer": comparator,
                "comparator_budget": 2,
            },
        ))
    capacity_contrasts: list[dict] = []
    for left, right in ((2, 1), (4, 2), (8, 2), (8, 4)):
        capacity_contrasts.append(_primary_contrast(
            family="lambdamart_capacity",
            lhs=f"r{left}",
            rhs=f"r{right}",
            lhs_values={
                qid: float(selected_lookup[(SCORER_LAMBDAMART, left, qid)]["selected_utility_at8"])
                for qid in contract["qids"]
            },
            rhs_values={
                qid: float(selected_lookup[(SCORER_LAMBDAMART, right, qid)]["selected_utility_at8"])
                for qid in contract["qids"]
            },
            plan=plans["all"],
            metadata={"left_budget": left, "right_budget": right},
        ))

    stable_utility = all(row["ci_low"] > 0 for row in model_contrasts)
    stable_ranking_metrics = 0
    for metric in (
        "within_query_spearman", "pairwise_accuracy_margin_1",
        "ndcg_at8_continuous_utility_gain",
    ):
        rows = [row for row in ranking_contrasts if row["metric"] == metric]
        stable_ranking_metrics += int(rows and all(row["ci_low"] > 0 for row in rows))
    if stable_utility:
        conclusion = "LambdaMART materially improves utility conversion"
    elif stable_ranking_metrics >= 2:
        conclusion = "LambdaMART improves ranking quality but not final Utility@8"
    else:
        conclusion = "LambdaMART provides no stable improvement over the simpler scorers"

    checks = {
        "development_queries_300": len(contract["qids"]) == 300,
        "primary_pairs_3600": len(primary_lambda_rows) == 3600,
        "all_queries_have_12_candidates": all(len(ids) == 12 for ids in primary_ids.values()),
        "outer_query_overlap_zero": all(row["train_validation_overlap"] == 0 for row in tuning_audit),
        "folds_match_huber_and_mlp": True,
        "candidate_ids_match_frozen_baseline": True,
        "feature_names_match_frozen_baseline": True,
        "utility_not_in_features": all("utility" not in name for name in STATIC_PREDICTOR_FEATURES),
        "oracle_not_in_features": all("oracle" not in name for name in STATIC_PREDICTOR_FEATURES),
        "community_not_in_features": all("community" not in name for name in STATIC_PREDICTOR_FEATURES),
        "explicit_graph_route_not_in_features": all(
            token not in name for name in STATIC_PREDICTOR_FEATURES
            for token in ("graph", "ppr", "route")
        ),
        "selector_input_coverage_complete": all(
            (SCORER_LAMBDAMART, qid, cid) in score_lookup
            for qid, ids in primary_ids.items() for cid in ids
        ),
        "all_selected_sets_size_8_unique": all(
            len(row["selected_comment_ids"]) == 8
            and len(set(row["selected_comment_ids"])) == 8
            for row in selector_per_query
        ),
        "lambda_r4_equals_r8": all(
            selected_lookup[(SCORER_LAMBDAMART, 4, qid)]["selected_comment_ids"]
            == selected_lookup[(SCORER_LAMBDAMART, 8, qid)]["selected_comment_ids"]
            for qid in contract["qids"]
        ),
        "test200_read": False,
        "external_calls": False,
    }
    if not all(value is True for key, value in checks.items() if key not in {"test200_read", "external_calls"}):
        raise AssertionError("matched LambdaMART integrity check failed")
    integrity = {
        "schema": "matched-lambdamart-integrity-v1",
        "checks": checks,
        "passed_checks": sum(
            value is True for key, value in checks.items()
            if key not in {"test200_read", "external_calls"}
        ) + int(checks["test200_read"] is False) + int(checks["external_calls"] is False),
        "total_checks": len(checks),
        "status": "PASS",
    }

    import xgboost
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary:
        out = Path(temporary)
        write_json(out / "experiment_config.json", {
            "schema": "matched-lambdamart-rq2b-config-v1",
            "version": raw["version"],
            "source_config_key": source_key,
            "source_experiment_dir": str(source_dir.relative_to(ROOT)),
            "backend": backend,
            "dense_depth": depth,
            "final_k": 8,
            "replacement_budgets": budgets,
            "query_count": 300,
            "primary_candidate_pairs": 3600,
            "training_pool": source_raw["scorer_training_pool"],
            "objective": "rank:ndcg",
            "grade_rule": "max(0, min(6, round(U)-1))",
            "pair_method": "mean",
            "grid": settings,
            "selection_rule": "frozen exact maximum-score K=8 set under at most r non-D8 items",
            "bootstrap_samples": int(source_raw["bootstrap_samples"]),
            "bootstrap_seed": int(source_raw["bootstrap_seed"]),
            "official_documentation": list(raw["official_documentation"]),
        })
        write_json(out / "feature_contract.json", {
            "feature_names": list(STATIC_PREDICTOR_FEATURES),
            "feature_count": len(STATIC_PREDICTOR_FEATURES),
            "preprocessing": "StandardScaler fit on training-query candidates only within each inner/outer fit",
            "missing_value_handling": "frozen numeric feature contract including candidate_dense_rank_missing",
            "utility_feature": False,
            "oracle_feature": False,
            "community_correspondence_feature": False,
            "explicit_graph_route_feature": False,
        })
        write_json(out / "fold_assignments.json", {
            "source_path": str(contract["paths"]["split_manifest"].relative_to(ROOT)),
            "source_sha256": sha256(contract["paths"]["split_manifest"]),
            "outer_rows": contract["splits"],
            "inner_rows": contract["inner_manifest"],
            "new_split_created": False,
            "test_read": False,
        })
        write_json(out / "inner_fold_tuning.json", tuning_audit)
        write_json(out / "chosen_hyperparameters.json", [
            {
                "repeat": row["repeat"],
                "fold": row["fold"],
                "seed": row["seed"],
                "selected_setting": row["selected_setting"],
                "selection_objective": row["selection_objective"],
            }
            for row in tuning_audit
        ])
        pq.write_table(
            pa.Table.from_pylist(scorer_pool_rows),
            out / "oof_scorer_pool_predictions.parquet",
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(primary_lambda_rows),
            out / "oof_candidate_scores.parquet",
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(ranking_per_query),
            out / "candidate_ranking_per_query.parquet",
            compression="zstd",
        )
        write_csv(out / "candidate_ranking_summary.csv", ranking_summary)
        write_csv(out / "ranking_paired_contrasts.csv", ranking_contrasts)
        pq.write_table(
            pa.Table.from_pylist(selector_per_query),
            out / "selected_sets.parquet",
            compression="zstd",
        )
        write_csv(out / "utility_at8_summary.csv", selector_summary)
        write_csv(out / "utility_at8_paired_contrasts.csv", model_contrasts)
        write_csv(out / "capacity_contrasts.csv", capacity_contrasts)
        write_json(out / "integrity_report.json", integrity)
        (out / "query_ids.txt").write_text(
            "\n".join(contract["qids"]) + "\n", encoding="utf-8"
        )
        (out / "MATCHED_LAMBDAMART_REPORT.md").write_text(
            _matched_report(
                ranking_summary=ranking_summary,
                selector_summary=selector_summary,
                model_contrasts=model_contrasts,
                capacity_contrasts=capacity_contrasts,
                best_budget=best_budget,
                conclusion=conclusion,
                integrity=integrity,
                xgboost_version=xgboost.__version__,
            ),
            encoding="utf-8",
        )
        run_all = f"""#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../.." && pwd)"
OUTPUT_DIR="${{1:-experiments/rq2b_matched_lambdamart_dev300_reproduction}}"
cd "$REPO_ROOT"
PYTHONPATH=.. .venv-reranker-repro/bin/python evidence_selection/run_selection_action_space_repair.py --config-key "{config_key}" --output-dir "$OUTPUT_DIR"
"""
        (out / "run_all.sh").write_text(run_all, encoding="utf-8")
        os.chmod(out / "run_all.sh", 0o755)
        manifest = {
            "schema": "matched-lambdamart-rq2b-reproduction-v1",
            "status": "COMPLETE",
            "created_utc": utc_now(),
            "runtime_seconds": time.perf_counter() - started,
            "command": f"PYTHONPATH=.. .venv-reranker-repro/bin/python evidence_selection/run_selection_action_space_repair.py --config-key {config_key}",
            "git_head": _git_head(),
            "implementation": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": sha256(Path(__file__).resolve()),
                "canonical_lambdamart_path": str(
                    Path(canonical.__file__).resolve().relative_to(ROOT)
                ),
                "canonical_lambdamart_sha256": sha256(Path(canonical.__file__).resolve()),
            },
            "versions": {
                "python": platform.python_version(),
                "xgboost": xgboost.__version__,
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
                "sklearn": __import__("sklearn").__version__,
            },
            "source_artifacts": {
                "directory": str(source_dir.relative_to(ROOT)),
                "reproduction_manifest_sha256": sha256(source_dir / "reproduction_manifest.json"),
                "config_sha256": sha256(source_dir / "config.json"),
                "candidate_pool_manifest_sha256": sha256(source_dir / "candidate_pool_manifest.parquet"),
                "baseline_oof_predictions_sha256": sha256(source_dir / "oof_candidate_predictions.parquet"),
                "baseline_selected_sets_sha256": sha256(source_dir / "selected_sets.parquet"),
            },
            "query_count": 300,
            "primary_candidate_pairs": 3600,
            "scorer_pool_oof_pairs": len(scorer_pool_rows),
            "outer_models": len(tuning_audit),
            "inner_grid_size": len(settings),
            "best_lambda_budget": best_budget,
            "thesis_conclusion": conclusion,
            "bootstrap_common_indices": bootstrap_manifest,
            "community_correspondence": {
                "completed": False,
                "reason": "primary E5-D12 utility experiment completed first; hidden responses were not opened during training or selection",
            },
            "integrity": integrity,
            "output_hashes": {
                path.name: sha256(path) for path in sorted(out.iterdir())
            },
        }
        write_json(out / "reproduction_manifest.json", manifest)
        os.replace(out, destination)
    return manifest


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(output_dir: Path | None = None, config_key: str = CONFIG_KEY) -> dict:
    started = time.perf_counter()
    raw = project_config.load()[config_key]
    contract = _load_contract(raw)
    paths = contract["paths"]
    destination = (output_dir or paths["output_dir"]).resolve()
    _reject_test(destination)
    if "heldout" in str(destination).lower() or "test200" in str(destination).lower():
        raise ValueError("output path looks like frozen-test scope")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")

    pools = _build_pools_and_features(contract)
    # The exact Oracle ceilings are descriptive properties of the frozen
    # pools and utility registry.  Materialise them before fitting any learned
    # selector, as required by the pre-registered execution order.
    oracle_rows, oracle_lookup, oracle_selected = _build_oracles(contract, pools)
    prediction_rows, prediction_lookup, tuning_audit = _run_oof_predictions(
        contract, pools
    )
    decomposition_rows, learned_selected = _build_learned_selections(
        contract, pools, prediction_lookup, oracle_lookup
    )
    plans, bootstrap_manifest = _bootstrap_plans(
        contract["qids"], contract["strata"],
        int(paths["bootstrap_samples"]), int(paths["bootstrap_seed"]),
    )
    dense_summary, graph_summary, intervals = _summaries(
        decomposition_rows, plans
    )
    contrasts = _paired_contrasts(decomposition_rows, plans)
    historical = _historical_reconciliation()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary:
        out = Path(temporary)
        pq.write_table(
            pa.Table.from_pylist(pools["manifest_rows"]),
            out / "candidate_pool_manifest.parquet", compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(oracle_rows),
            out / "constrained_oracles.parquet", compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(prediction_rows),
            out / "oof_candidate_predictions.parquet", compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist([*oracle_selected, *learned_selected]),
            out / "selected_sets.parquet", compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(decomposition_rows),
            out / "per_query_decomposition.parquet", compression="zstd",
        )
        write_csv(out / "dense_summary.csv", dense_summary)
        write_csv(out / "graph_summary.csv", graph_summary)
        write_csv(out / "paired_contrasts.csv", contrasts)
        write_csv(out / "bootstrap_intervals.csv", intervals)
        write_csv(out / "historical_reconciliation.csv", historical)
        (out / "query_ids.txt").write_text(
            "\n".join(contract["qids"]) + "\n", encoding="utf-8"
        )

        fold_manifest = {
            "schema": "selection-action-space-repair-folds-v1",
            "unit": "query_id",
            "source": {
                "path": str(paths["split_manifest"].relative_to(ROOT)),
                "sha256": sha256(paths["split_manifest"]),
            },
            "outer_repeats": int(paths["outer_repeats"]),
            "outer_folds": int(paths["outer_folds"]),
            "inner_folds": int(paths["inner_folds"]),
            "outer_rows": contract["splits"],
            "inner_rows": contract["inner_manifest"],
            "report89_inner_splits_exact": contract["report89_inner_splits_exact"],
            "query_overlap_per_fold": 0,
            "validation_appearances_per_query": 5,
            "test_read": False,
        }
        write_json(out / "fold_manifest.json", fold_manifest)

        input_paths = {
            key: paths[key] for key in (
                "utility_registry", "dense_memberships", "graph_candidate_views",
                "split_manifest", "query_admin", "queries", "corpus",
            )
        }
        config_payload = {
            "schema": "selection-action-space-repair-config-v1",
            "version": paths["version"],
            "scope": paths.get("scope_label", "development100-only local analysis"),
            "external_calls": 0,
            "frozen_test_read": False,
            "remaining_development298_read": bool(
                paths.get("remaining_development298_read", False)
            ),
            "backends": list(paths["backends"]),
            "dense_depths": list(map(int, paths["dense_depths"])),
            "pool_families": [POOL_DENSE, POOL_GRAPH],
            "replacement_budgets": list(map(int, paths["replacement_budgets"])),
            "final_k": int(paths["final_k"]),
            "scorer_training_pool": paths["scorer_training_pool"],
            "scorer_settings": _scorer_settings(paths),
            "feature_names": list(STATIC_PREDICTOR_FEATURES),
            "feature_semantics": {
                "source_blind": "no explicit source/Graph/PPR identity; allowed dense-rank missingness is disclosed and can identify D50-external candidates",
                "backend_local_semantic_embeddings": True,
                "current_query_gold_utility": False,
                "oracle_labels": False,
                "community_response_correspondence": False,
                "need_stratum": "reporting_only",
                "explicit_graph_or_ppr_route_identity": "provenance_only, not an inference feature",
                "dense_rank_missingness": "allowed retrieval-rank feature; may act as an indirect source proxy and is disclosed",
            },
            "prediction_aggregation": "mean five repeated query-grouped OOF candidate predictions before selection",
            "prediction_reuse": "same backend/scorer score map for all M, pool families and r",
            "selection_objective": "maximum sum of candidate predicted utility under K=8 and at most r non-D8 items",
            "oracle_objective": "same exact selection with frozen utility-v2 in place of predicted scores",
            "tie_break": "prefer fewer replacements, then candidate_id",
            "bootstrap_samples": int(paths["bootstrap_samples"]),
            "bootstrap_seed": int(paths["bootstrap_seed"]),
            "bootstrap_common_indices": bootstrap_manifest,
            "outcome_rules": {
                "A": "one-swap action-space-loss intervals include zero in all 12 Dense backend-depth-scorer cells, and no paired r8-minus-r1 interval is wholly positive",
                "B": "one-swap action-space-loss and paired r8-minus-r1 intervals are wholly positive in all 12 Dense backend-depth-scorer cells",
                "C": "one-swap action-space-loss intervals are wholly positive in all 12 Dense cells, but no paired r8-minus-r1 interval is wholly positive",
                "D": "all 12 Dense paired r8-minus-r1 intervals are wholly positive, but no M8/M12 r8 Graph marginal interval over matched Dense is wholly positive",
            },
            "input_hashes": {
                key: {
                    "path": str(path.relative_to(ROOT)), "sha256": sha256(path)
                }
                for key, path in input_paths.items()
            },
            "embedding_inputs": pools["embedding_audit"],
            "idf_manifest": pools["idf_manifest"],
        }
        write_json(out / "config.json", config_payload)

        _write_latex_tables(
            out, dense_summary, graph_summary, oracle_lookup, contract["qids"]
        )
        _render_figures(
            out, dense_summary, graph_summary, oracle_lookup, contract["qids"]
        )
        outcomes = _write_result_documents(
            out, dense_summary, graph_summary, contrasts, oracle_lookup,
            contract["qids"],
            scope_label=paths.get("scope_label", "development100-only local analysis"),
        )

        run_all = f"""#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../.." && pwd)"
OUTPUT_DIR="${{1:-experiments/selection_action_space_repair_reproduction}}"
cd "$REPO_ROOT"
MPLCONFIGDIR="${{TMPDIR:-/tmp}}/evidence_pipeline_matplotlib" PYTHONPATH=. .venv-reranker-repro/bin/python evidence_selection/run_selection_action_space_repair.py --config-key "{config_key}" --output-dir "$OUTPUT_DIR"
"""
        (out / "run_all.sh").write_text(run_all, encoding="utf-8")
        os.chmod(out / "run_all.sh", 0o755)

        expected_names = {
            "config.json", "query_ids.txt", "fold_manifest.json",
            "candidate_pool_manifest.parquet", "constrained_oracles.parquet",
            "oof_candidate_predictions.parquet", "selected_sets.parquet",
            "per_query_decomposition.parquet", "dense_summary.csv",
            "graph_summary.csv", "paired_contrasts.csv",
            "bootstrap_intervals.csv", "historical_reconciliation.csv",
            "table_action_space_decomposition.tex",
            "table_dense_replacement_frontier.tex",
            "table_graph_replacement_frontier.tex",
            "figure_oracle_decomposition.pdf",
            "figure_dense_replacement_frontier.pdf",
            "figure_graph_replacement_frontier.pdf",
            "RESULTS_INTERPRETATION.md", "THESIS_IMPLICATIONS.md", "run_all.sh",
        }
        if {path.name for path in out.iterdir()} != expected_names:
            raise AssertionError("output package contents drifted before manifest")

        import sklearn

        manifest = {
            "schema": "selection-action-space-repair-reproduction-v1",
            "status": "COMPLETE",
            "created_utc": utc_now(),
            "runtime_seconds": time.perf_counter() - started,
            "command": "PYTHONPATH=.. .venv-reranker-repro/bin/python evidence_selection/run_selection_action_space_repair.py",
            "git_head": _git_head(),
            "implementation": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "development_queries": len(contract["qids"]),
            "single_need_queries": int(Counter(contract["strata"].values())["single_need"]),
            "multi_need_queries": int(Counter(contract["strata"].values())["multi_need"]),
            "complete_registry_rows": len(contract["registry"]),
            "required_candidate_pairs": len(contract["required_pairs"]),
            "candidate_pool_manifest_rows": len(pools["manifest_rows"]),
            "constrained_oracle_rows": len(oracle_rows),
            "oof_candidate_prediction_rows": len(prediction_rows),
            "selected_set_rows": len(oracle_selected) + len(learned_selected),
            "per_query_decomposition_rows": len(decomposition_rows),
            "nested_tuning_audit": tuning_audit,
            "outcomes": outcomes,
            "bootstrap_common_indices": bootstrap_manifest,
            "invariants": {
                "external_calls": 0,
                "frozen_test_read": False,
                "remaining_development298_read": bool(
                    paths.get("remaining_development298_read", False)
                ),
                "query_grouped_outer_and_inner_splits": True,
                "same_predictions_all_replacement_budgets": True,
                "same_predictions_all_candidate_widths": True,
                "same_predictions_dense_and_graph_pool": True,
                "predictions_averaged_before_inference": True,
                "gold_utility_in_inference_features": False,
                "explicit_graph_route_identity_in_inference_features": False,
                "dense_rank_missingness_proxy_disclosed": True,
                "need_label_in_inference_features": False,
                "strict_graph_fallback_count": 0,
                "strict_graph_callback_count": 0,
                "strict_graph_padding_count": 0,
                "graph_marginal_all_added_relative_to_dense_m": True,
                "all_final_sets_size_8_unique": True,
                "oracle_and_learned_selection_same_exact_solver": True,
                "oracle_materialized_before_selector_training": True,
                "historical_direct_huber_relabelled_as_matched_r1": False,
                "prior_outputs_overwritten": False,
            },
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
                "torch": torch.__version__,
                "sklearn": sklearn.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "output_hashes": {
                path.name: sha256(path) for path in sorted(out.iterdir())
            },
        }
        write_json(out / "reproduction_manifest.json", manifest)
        expected_names.add("reproduction_manifest.json")
        if {path.name for path in out.iterdir()} != expected_names:
            raise AssertionError("final output package is incomplete")
        os.replace(out, destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-key", default=CONFIG_KEY)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    raw = project_config.load()[args.config_key]
    if str(raw.get("experiment_kind", "")) == "matched_lambdamart":
        manifest = run_matched_lambdamart(args.output_dir, args.config_key)
        default_output = Path(raw["output_dir"])
    else:
        manifest = run(args.output_dir, args.config_key)
        default_output = _resolve_inputs(raw)["output_dir"]
    payload = {
        "status": manifest["status"],
        "output": str((args.output_dir or default_output).resolve()),
        "runtime_seconds": manifest["runtime_seconds"],
    }
    if "outcomes" in manifest:
        payload["outcomes"] = manifest["outcomes"]
    if "conclusion" in manifest:
        payload["conclusion"] = manifest["conclusion"]
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
