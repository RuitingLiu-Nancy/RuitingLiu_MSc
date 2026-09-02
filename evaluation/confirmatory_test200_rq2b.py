"""RQ2b Test200 implementation used by the canonical confirmatory runner.

This module deliberately has no CLI.  ``run_confirmatory_test200.py`` remains
the single entry point and dispatches here only for the preregistered RQ2b
confirmation.  The final LambdaMART model is fitted and content-addressed in a
Development300-only phase before any Test200 artefact is opened.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

import configuration as project_config
from evaluation.judgment_completeness import DIMS_V2
from fusion.ranking import cc_scores, rrf_scores
from candidate_pool.analyze_strict_sbert_graph_oracle import (
    _round_robin_graph_head,
)
from candidate_pool.run_dense_semantic_drift_rescue_audit import _build_idf
from candidate_pool.run_m50_dense_frontier_analysis import (
    STATIC_PREDICTOR_FEATURES,
    static_features_for_arm,
)
from evidence_selection.run_selection_action_space_repair import (
    SCORER_HUBER,
    _build_pools_and_features,
    _load_contract,
)


PROTOCOL = "confirmatory-test200-rq2b-lambdamart-v2"
ARMS = (
    "t0_dense_top8",
    "t1_d12_one_swap_huber",
    "t2_m50_lambdamart_direct",
    "t3_m50_lambdamart_residual_beta0175",
)
BACKENDS = ("e5",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_lines(values: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in values).encode("utf-8")
    ).hexdigest()


def _rooted(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _top_ids(scores: dict[str, float], k: int) -> list[str]:
    return sorted(scores, key=lambda candidate_id: (-float(scores[candidate_id]), candidate_id))[:k]


def _rank_prior_fused_scores(
    entry_ids: list[str],
    entry_scores: dict[str, float],
    model_scores: dict[str, float],
    *,
    mode: str,
    entry_weight: float,
    k0: int,
    normalization: str,
) -> dict[str, float]:
    if set(entry_ids) != set(model_scores) or set(entry_ids) != set(entry_scores):
        raise ValueError("rank-prior fusion inputs do not share candidate identities")
    entry_run = [
        {"comment_id": candidate_id, "score": float(entry_scores[candidate_id])}
        for candidate_id in entry_ids
    ]
    model_ids = sorted(model_scores, key=lambda candidate_id: (-model_scores[candidate_id], candidate_id))
    model_run = [
        {"comment_id": candidate_id, "score": float(model_scores[candidate_id])}
        for candidate_id in model_ids
    ]
    runs = {"entry": entry_run, "model": model_run}
    weights = {"entry": float(entry_weight), "model": 1.0 - float(entry_weight)}
    if mode == "rank_rrf":
        fused = rrf_scores(runs, weights=weights, k0=k0)
    elif mode == "score_cc":
        fused = cc_scores(runs, weights=weights, normalization=normalization)
    else:
        raise ValueError(f"unknown rank-prior fusion mode: {mode}")
    if set(fused) != set(entry_ids):
        raise AssertionError("rank-prior fusion dropped a candidate")
    return dict(fused)


def _load_flat_rankings(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(path):
        grouped[str(row["query_id"])].append(row)
    result = {}
    for query_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["rank"]))
        identities = {str(row["comment_id"]) for row in rows}
        if len(rows) < 50 or len(identities) != len(rows):
            raise ValueError(f"{path}/{query_id}: invalid ranking")
        result[query_id] = rows
    return result


def _modal_setting(settings: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], int]:
    counts: Counter[str] = Counter()
    payloads: dict[str, dict[str, Any]] = {}
    for setting in settings:
        key = json.dumps(dict(setting), sort_keys=True, separators=(",", ":"))
        counts[key] += 1
        payloads[key] = dict(setting)
    if not counts:
        raise ValueError("no fitted hyperparameter settings were supplied")
    best_count = max(counts.values())
    winner = sorted(key for key, count in counts.items() if count == best_count)[0]
    return payloads[winner], best_count


def _frozen_hyperparameters(cfg: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    huber_manifest = _read_json(Path(cfg["huber_training_manifest"]))
    huber_settings = [
        row["selected_setting"]
        for row in huber_manifest["nested_tuning_audit"]
        if str(row["backend"]) == "e5" and str(row["scorer"]) == SCORER_HUBER
    ]
    huber, huber_count = _modal_setting(huber_settings)
    lambda_rows = _read_json(Path(cfg["lambdamart_hyperparameters"]))
    lambdamart, lambda_count = _modal_setting(
        row["selected_setting"] for row in lambda_rows
    )
    if huber_count != 25 or lambda_count < 13:
        raise ValueError("modal hyperparameter support changed")
    return huber, lambdamart, {
        "huber_modal_outer_models": huber_count,
        "huber_outer_models": len(huber_settings),
        "lambdamart_modal_outer_models": lambda_count,
        "lambdamart_outer_models": len(lambda_rows),
    }


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _assert_graph_post_disjoint(test_rows: list[dict], source: Path) -> None:
    with source.open(encoding="utf-8", newline="") as handle:
        graph_posts = {
            str(row.get("post_id") or row.get("query_id") or "")
            for row in csv.DictReader(handle)
        }
    overlap = {str(row["post_id"]) for row in test_rows} & graph_posts
    if overlap:
        raise ValueError(f"held-out posts occur in graph extraction: {len(overlap)}")


def _load_route(
    path: Path,
    *,
    require_native_entry: bool = True,
    expected_query_ids: list[str] | None = None,
) -> dict[str, list[dict]]:
    expected = set(expected_query_ids or [])
    native_entry = {}
    if require_native_entry:
        trace_path = path.with_suffix(".entry_trace.jsonl")
        if not trace_path.exists():
            raise FileNotFoundError(f"graph route lacks entry trace: {trace_path}")
        trace_rows = _read_jsonl(trace_path)
        trace_ids = [str(row["query_id"]) for row in trace_rows]
        if len(trace_ids) != len(set(trace_ids)):
            raise ValueError(f"graph entry trace has duplicate queries: {path}")
        if expected and set(trace_ids) != expected:
            raise ValueError(f"graph entry trace query set mismatch: {path}")
        native_entry = {
            query_id: int(row.get("selected_fact_count") or 0) > 0
            for query_id, row in zip(trace_ids, trace_rows, strict=True)
        }
    route_rows = _read_jsonl(path)
    route_ids = [str(row["query_id"]) for row in route_rows]
    if len(route_ids) != len(set(route_ids)):
        raise ValueError(f"graph route has duplicate queries: {path}")
    if expected and set(route_ids) != expected:
        raise ValueError(f"graph route query set mismatch: {path}")
    result = {}
    for row in route_rows:
        query_id = str(row["query_id"])
        titles = list(map(str, row["retrieved_titles"]))
        scores = list(map(float, row["retrieved_scores"]))
        if len(titles) != len(scores):
            raise ValueError(f"route length mismatch: {query_id}")
        ranked = [
            {"comment_id": candidate_id, "rank": rank, "score": score}
            for rank, (candidate_id, score) in enumerate(zip(titles, scores, strict=True), 1)
            if math.isfinite(score) and score > 0
        ]
        result[query_id] = ranked if not require_native_entry or native_entry.get(query_id) else []
    return result


def _load_verified_judging_preflight(cfg: dict) -> tuple[Path, dict, list, list]:
    preflight = Path(cfg["output_dir"]) / str(cfg["judging_preflight_subdir"])
    manifest_path = preflight / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("utility judging preflight is absent")
    manifest = _read_json(manifest_path)
    authorization = dict(cfg.get("external_authorization") or {})
    if set(authorization.get("residual_gate_verdicts") or []) != {
        "STABLE", "STABLE_WITH_MINOR_DRIFT"
    }:
        raise ValueError("residual anchor gate changed")
    if manifest.get("authorization", {}).get("record") != authorization:
        raise ValueError("external authorization record changed after preflight")
    configured_expected = cfg.get("expected_judging_residual_pairs")
    expected_residual = (
        int(manifest.get("residual_unique_pairs", -1))
        if configured_expected in (None, "auto")
        else int(configured_expected)
    )
    required = {
        "phase": manifest.get("phase") == "prepare-judging-complete",
        "scope": manifest.get("judging_scope") == "final_system_output_union",
        "residual_count": manifest.get("residual_unique_pairs") == expected_residual,
        "anchors": manifest.get("calibration_anchors") == 50,
        "authorized": manifest.get("authorization", {}).get("authorised_at_preflight") is True,
        "external_calls_before_execution": manifest.get("external_calls") == 0,
        "test_outcomes_not_read": manifest.get("test_outcomes_read") is False,
    }
    if not all(required.values()):
        raise ValueError(f"utility judging preflight gate failed: {required}")
    for name, expected_hash in manifest["prepared_file_sha256"].items():
        artifact = preflight / name
        if not artifact.exists() or _sha256(artifact) != expected_hash:
            raise ValueError(f"utility judging preflight artifact drift: {name}")
    if _sha256(cfg["judge_prompt"]) != manifest["judge_protocol"]["prompt_sha256"]:
        raise ValueError("utility prompt changed after preflight")
    for relative, expected_hash in manifest["critical_local_output_sha256"].items():
        artifact = Path(cfg["output_dir"]) / relative
        if not artifact.exists() or _sha256(artifact) != expected_hash:
            raise ValueError(f"system artifact drift: {relative}")
    payloads = _read_jsonl(preflight / "residual_payload.jsonl")
    admin = _read_jsonl(preflight / "residual_payload_admin.jsonl")
    anchors = _read_jsonl(preflight / "anchor_payload.jsonl")
    anchor_admin = _read_jsonl(preflight / "anchor_payload_admin.jsonl")
    if len(payloads) != len(admin) or len(anchors) != len(anchor_admin):
        raise ValueError("provider payload and local mapping lengths differ")
    for row, payload in zip(admin, payloads, strict=True):
        if set(payload) != {"query_text", "comment_text", "facets_json"}:
            raise ValueError("residual provider payload fields changed")
        if payload["facets_json"] != {} or _hash_json(payload) != row["provider_payload_sha256"]:
            raise ValueError("residual payload integrity check failed")
    for row, payload in zip(anchor_admin, anchors, strict=True):
        if set(payload) != {"query_text", "comment_text", "facets_json"}:
            raise ValueError("anchor provider payload fields changed")
        if _hash_json(payload) != row["provider_payload_sha256"]:
            raise ValueError("anchor payload integrity check failed")
    return preflight, manifest, [admin, payloads], [anchor_admin, anchors]


def _paired_bootstrap_ci(
    deltas: list[float], *, seed: int = 20260731, draws: int = 10_000
) -> tuple[float, float]:
    values = np.asarray(deltas, dtype=float)
    if not len(values):
        raise ValueError("paired bootstrap received no queries")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for start in range(0, draws, 1000):
        stop = min(start + 1000, draws)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def resolve_config(root: Path, raw: dict) -> dict:
    cfg = dict(raw)
    path_keys = (
        "output_dir", "final_model_dir", "test_ids", "test_admin",
        "partition_manifest", "corpus", "graph_extraction_posts",
        "e5_corpus_embeddings", "test_e5_query_embeddings",
        "test_e5_rankings", "test_minilm_rankings", "historical_system_rankings",
        "existing_test_utility_registry", "huber_training_manifest",
        "lambdamart_hyperparameters", "lambdamart_runtime", "anchor_payload",
        "anchor_payload_admin", "pricing_precheck", "judge_prompt",
        "preregistration",
    )
    for key in path_keys:
        cfg[key] = _rooted(root, cfg[key])
    cfg["test_graph_routes"] = {
        name: _rooted(root, path)
        for name, path in cfg["test_graph_routes"].items()
    }
    cfg["test_graph_entry_traces"] = {
        name: _rooted(root, path)
        for name, path in cfg["test_graph_entry_traces"].items()
    }
    return cfg


def _final_model_manifest(cfg: Mapping[str, Any]) -> tuple[Path, dict]:
    directory = Path(cfg["final_model_dir"])
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "P1 is incomplete: run freeze-final-model before any Test200 phase"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "FROZEN_BEFORE_TEST_CONTACT":
        raise ValueError("final LambdaMART freeze manifest is not complete")
    for name, expected in manifest["outputs"].items():
        path = directory / name
        if not path.exists() or _sha256(path) != expected:
            raise ValueError(f"frozen final-model artefact changed: {name}")
    return directory, manifest


def _training_arrays(
    cfg: Mapping[str, Any],
) -> tuple[dict, dict, list[tuple[str, str]], np.ndarray, np.ndarray]:
    source_raw = project_config.load()[str(cfg["development300_source_config_key"])]
    contract = _load_contract(source_raw)
    pools = _build_pools_and_features(contract)
    pairs = [
        (qid, cid)
        for qid in contract["qids"]
        for cid in pools["max_pool_ids"]["e5"][qid]
    ]
    train_x = np.asarray(
        [
            [
                float(pools["features"]["e5"][pair][name])
                for name in STATIC_PREDICTOR_FEATURES
            ]
            for pair in pairs
        ],
        dtype=np.float64,
    )
    train_y = np.asarray(
        [float(contract["registry"][pair]["utility"]) for pair in pairs],
        dtype=np.float64,
    )
    return contract, pools, pairs, train_x, train_y


def freeze_final_model(cfg: dict) -> dict:
    """P1: fit once on Development300 and save the exact transfer model."""
    destination = Path(cfg["final_model_dir"])
    if destination.exists():
        _, manifest = _final_model_manifest(cfg)
        return manifest
    if bool(cfg.get("test_contacted_before_model_freeze")):
        raise PermissionError("configuration records Test contact before model freeze")
    contract, _, pairs, train_x, train_y = _training_arrays(cfg)
    _, setting, support = _frozen_hyperparameters(cfg)
    seed = int(cfg["final_model_seed"])
    runtime = Path(cfg["lambdamart_runtime"])
    if not runtime.exists():
        raise FileNotFoundError(runtime)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="rq2b_final_lambda_", dir=destination.parent))
    try:
        bundle = staging / "training_bundle.npz"
        setting_path = staging / "setting.json"
        predictions = staging / "training_predictions.npy"
        model_path = staging / "lambdamart_model.json"
        scaler_path = staging / "standard_scaler.npz"
        np.savez_compressed(
            bundle,
            train_x=train_x,
            train_y=train_y,
            train_qids=np.asarray([qid for qid, _ in pairs], dtype=str),
            predict_x=train_x,
        )
        _write_json(setting_path, setting)
        command = [
            str(runtime),
            str(Path(__file__).with_name("fit_lambdamart_transfer.py")),
            str(bundle), str(setting_path), str(predictions),
            "--seed", str(seed),
            "--model-output", str(model_path),
            "--scaler-output", str(scaler_path),
        ]
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "MPLCONFIGDIR": "/tmp/graphrag-mpl-cache"},
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "final LambdaMART freeze failed: "
                + (completed.stderr or completed.stdout)[-4000:]
            )
        training_contract = {
            "development_queries": len(contract["qids"]),
            "training_rows": len(pairs),
            "training_query_order_sha256": _hash_lines(list(contract["qids"])),
            "training_pair_order_sha256": _hash_lines(
                [f"{qid}\t{cid}" for qid, cid in pairs]
            ),
            "feature_names": list(STATIC_PREDICTOR_FEATURES),
            "feature_count": len(STATIC_PREDICTOR_FEATURES),
            "utility_role": "utility-v2 LLM simulated-user silver; not human gold",
            "lambdamart_setting": setting,
            "hyperparameter_support": support,
            "seed": seed,
            "runtime": str(runtime),
            "runtime_stdout": completed.stdout.strip(),
        }
        _write_json(staging / "training_contract.json", training_contract)
        manifest = {
            "schema": "rq2b-final-lambdamart-dev300-freeze-v1",
            "status": "FROZEN_BEFORE_TEST_CONTACT",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "test_split_opened": False,
            "test_ids_opened": False,
            "test_admin_opened": False,
            "external_calls": 0,
            "training": training_contract,
            "outputs": {
                name: _sha256(staging / name)
                for name in (
                    "lambdamart_model.json", "standard_scaler.npz",
                    "setting.json", "training_contract.json",
                    "training_predictions.npy",
                )
            },
            "definition_provenance": {
                "training_contract": str(cfg["development300_source_config_key"]),
                "fit_function": "tools.learned_diffusion.reranker_validation.fit_xgb_lambdamart",
                "runtime_bridge": "utility_scoring/fit_lambdamart_transfer.py",
                "feature_ssot": "STATIC_PREDICTOR_FEATURES",
            },
        }
        _write_json(staging / "manifest.json", manifest)
        bundle.unlink()
        os.replace(staging, destination)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _read_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = [str(row.get("query_id") or row.get("post_id") or "") for row in rows]
    if not values or any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError("frozen Test ID file is invalid")
    return values


def verify_contract_rq2b(cfg: dict) -> dict:
    _, model_manifest = _final_model_manifest(cfg)
    if tuple(map(str, cfg["arms"])) != ARMS or tuple(map(str, cfg["backends"])) != BACKENDS:
        raise ValueError("RQ2b Test200 arm/backend contract changed")
    if (
        int(cfg["pool_depth"]) != 50
        or int(cfg["final_k"]) != 8
        or float(cfg["residual_beta"]) != 0.175
        or int(cfg["rrf_k0"]) != 60
        or float(cfg["dense_weight"]) != 1.0
        or float(cfg["graph_weight"]) != 0.3
    ):
        raise ValueError("preregistered RQ2b Test200 parameters changed")
    required = [
        cfg["test_ids"], cfg["test_admin"], cfg["partition_manifest"],
        cfg["corpus"], cfg["graph_extraction_posts"], cfg["e5_corpus_embeddings"],
        cfg["test_e5_query_embeddings"], cfg["test_e5_rankings"],
        cfg["test_minilm_rankings"], cfg["historical_system_rankings"],
        cfg["existing_test_utility_registry"], cfg["preregistration"],
        *cfg["test_graph_routes"].values(),
        *cfg["test_graph_entry_traces"].values(),
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen RQ2b Test200 inputs: {missing}")
    ids = _read_ids(Path(cfg["test_ids"]))
    if len(ids) != int(cfg["expected_test_count"]):
        raise ValueError("frozen Test200 count changed")
    checks = {
        "test_ids": _sha256(Path(cfg["test_ids"])),
        "corpus": _sha256(Path(cfg["corpus"])),
    }
    if checks["test_ids"] != str(cfg["expected_test_ids_sha256"]):
        raise ValueError("frozen Test ID file changed")
    if checks["corpus"] != str(cfg["expected_corpus_sha256"]):
        raise ValueError("frozen corpus changed")
    frozen = {
        "test_e5_query_embeddings": Path(cfg["test_e5_query_embeddings"]),
        "test_e5_rankings": Path(cfg["test_e5_rankings"]),
        "test_minilm_rankings": Path(cfg["test_minilm_rankings"]),
        "historical_system_rankings": Path(cfg["historical_system_rankings"]),
        "existing_test_utility_registry": Path(cfg["existing_test_utility_registry"]),
        **{
            f"graph_route_{name}": Path(path)
            for name, path in cfg["test_graph_routes"].items()
        },
        **{
            f"graph_trace_{name}": Path(path)
            for name, path in cfg["test_graph_entry_traces"].items()
        },
    }
    expected = dict(cfg["frozen_test_artifact_sha256"])
    actual = {name: _sha256(path) for name, path in frozen.items()}
    if set(actual) != set(expected) or any(actual[name] != expected[name] for name in actual):
        raise ValueError("frozen Test200 supporting artefact changed")
    authorization = dict(cfg.get("external_authorization") or {})
    if cfg.get("allow_external_judging") and authorization.get("granted") is not True:
        raise ValueError("external judging lacks explicit recorded authorization")
    return {
        "protocol": PROTOCOL,
        "phase": "verify",
        "p1_final_model_frozen_before_test": True,
        "final_model_manifest_sha256": _sha256(Path(cfg["final_model_dir"]) / "manifest.json"),
        "final_model_outputs": dict(model_manifest["outputs"]),
        "test_split_used": False,
        "test_admin_opened": False,
        "external_calls": 0,
        "test_count": len(ids),
        "test_ids_sha256": checks["test_ids"],
        "test_id_set_sha256": hashlib.sha256(
            "".join(f"{value}\n" for value in sorted(ids)).encode("utf-8")
        ).hexdigest(),
        "expected_test_admin_sha256": str(cfg["expected_test_admin_sha256"]),
        "corpus_sha256": checks["corpus"],
        "frozen_test_artifact_sha256": actual,
        "arms": list(ARMS),
        "backends": list(BACKENDS),
        "external_authorization_recorded": authorization.get("granted") is True,
    }


def _predict_frozen_model(
    cfg: Mapping[str, Any], pairs: list[tuple[str, str]], features: Mapping,
    staging: Path,
) -> tuple[dict[tuple[str, str], float], dict]:
    model_dir, model_manifest = _final_model_manifest(cfg)
    predict_x = np.asarray(
        [
            [float(features[pair][name]) for name in STATIC_PREDICTOR_FEATURES]
            for pair in pairs
        ],
        dtype=np.float64,
    )
    bundle = staging / "PRIVATE" / "test_prediction_features.npz"
    predictions_path = staging / "PRIVATE" / "test_lambdamart_predictions.npy"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(bundle, predict_x=predict_x)
    command = [
        str(cfg["lambdamart_runtime"]),
        str(Path(__file__).with_name("fit_lambdamart_transfer.py")),
        str(bundle), str(model_dir / "setting.json"), str(predictions_path),
        "--seed", str(cfg["final_model_seed"]), "--mode", "predict",
        "--model-input", str(model_dir / "lambdamart_model.json"),
        "--scaler-input", str(model_dir / "standard_scaler.npz"),
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "MPLCONFIGDIR": "/tmp/graphrag-mpl-cache"},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "frozen LambdaMART prediction failed: "
            + (completed.stderr or completed.stdout)[-4000:]
        )
    values = np.load(predictions_path, allow_pickle=False)
    if len(values) != len(pairs):
        raise ValueError("frozen LambdaMART prediction coverage changed")
    return dict(zip(pairs, map(float, values), strict=True)), {
        "runtime_stdout": completed.stdout.strip(),
        "prediction_rows": len(pairs),
        "feature_names": list(STATIC_PREDICTOR_FEATURES),
        "final_model_manifest_sha256": _sha256(model_dir / "manifest.json"),
        "final_model_outputs": dict(model_manifest["outputs"]),
        "model_refit_after_test_contact": False,
    }


def prepare_local_rq2b(cfg: dict, *, confirm_test_read: bool) -> dict:
    contract = verify_contract_rq2b(cfg)
    if not confirm_test_read or not bool(cfg.get("allow_frozen_test_read")):
        raise PermissionError("prepare-local requires the explicit Test200 read gate")
    output = Path(cfg["output_dir"])
    if output.exists():
        raise FileExistsError(f"versioned output already exists: {output}")
    if _sha256(Path(cfg["test_admin"])) != str(cfg["expected_test_admin_sha256"]):
        raise ValueError("frozen Test ADMIN changed")
    with Path(cfg["test_admin"]).open(encoding="utf-8", newline="") as handle:
        test_rows = list(csv.DictReader(handle))
    test_ids = _read_ids(Path(cfg["test_ids"]))
    if [str(row["query_id"]) for row in test_rows] != test_ids:
        raise ValueError("Test ADMIN order differs from frozen Test IDs")
    _assert_graph_post_disjoint(test_rows, Path(cfg["graph_extraction_posts"]))
    staging = Path(tempfile.mkdtemp(prefix="confirmatory_rq2b_", dir=output.parent))
    try:
        corpus_rows = _read_json(Path(cfg["corpus"]))
        corpus_ids = [str(row["title"]) for row in corpus_rows]
        corpus_text = {str(row["title"]): str(row["text"]) for row in corpus_rows}
        query_text = {str(row["query_id"]): str(row["query_text"]) for row in test_rows}
        e5 = _load_flat_rankings(Path(cfg["test_e5_rankings"]))
        minilm = _load_flat_rankings(Path(cfg["test_minilm_rankings"]))
        if set(e5) != set(test_ids) or set(minilm) != set(test_ids):
            raise ValueError("frozen Test ranking query coverage changed")
        routes = {
            name: _load_route(Path(path), expected_query_ids=test_ids)
            for name, path in cfg["test_graph_routes"].items()
        }
        d8 = {
            qid: [str(row["comment_id"]) for row in e5[qid][:8]]
            for qid in test_ids
        }
        d50 = {
            qid: [str(row["comment_id"]) for row in e5[qid][:50]]
            for qid in test_ids
        }
        graph50: dict[str, list[str]] = {}
        entry_ids: dict[str, list[str]] = {}
        entry_scores: dict[str, dict[str, float]] = {}
        p3_rows = []
        for qid in test_ids:
            graph50[qid] = _round_robin_graph_head(
                {
                    name: [str(row["comment_id"]) for row in run[qid]]
                    for name, run in routes.items()
                },
                {str(row["comment_id"]) for row in minilm[qid][:8]},
                int(cfg["graph_budget"]),
            )
            if len(graph50[qid]) != int(cfg["graph_budget"]):
                raise ValueError(f"{qid}: strict-native G50 could not be filled")
            fused = rrf_scores(
                {
                    "dense": [
                        {"comment_id": str(row["comment_id"]), "score": float(row["score"])}
                        for row in e5[qid][:50]
                    ],
                    "graph": [
                        {"comment_id": cid, "score": 1.0 / rank}
                        for rank, cid in enumerate(graph50[qid], 1)
                    ],
                },
                weights={
                    "dense": float(cfg["dense_weight"]),
                    "graph": float(cfg["graph_weight"]),
                },
                k0=int(cfg["rrf_k0"]),
            )
            entry_ids[qid] = sorted(
                fused, key=lambda cid: (-float(fused[cid]), cid)
            )[: int(cfg["pool_depth"])]
            entry_scores[qid] = {
                cid: float(fused[cid]) for cid in entry_ids[qid]
            }
            p3_rows.append({
                "query_id": qid,
                "candidate_set_identical": set(entry_ids[qid]) == set(d50[qid]),
                "candidate_order_identical": entry_ids[qid] == d50[qid],
                "rrf_only_count": len(set(entry_ids[qid]) - set(d50[qid])),
                "dense_only_count": len(set(d50[qid]) - set(entry_ids[qid])),
                "top8_overlap": len(set(entry_ids[qid][:8]) & set(d8[qid])),
            })

        corpus_vectors_array = np.load(Path(cfg["e5_corpus_embeddings"]), mmap_mode="r")
        query_vectors_array = np.load(Path(cfg["test_e5_query_embeddings"]), mmap_mode="r")
        if len(corpus_vectors_array) != len(corpus_ids) or len(query_vectors_array) != len(test_ids):
            raise ValueError("frozen E5 vector/text identity changed")
        candidate_vectors = dict(zip(corpus_ids, corpus_vectors_array, strict=True))
        query_vectors = dict(zip(test_ids, query_vectors_array, strict=True))
        rank_maps = {
            qid: {
                str(row["comment_id"]): int(row["rank"])
                for row in e5[qid]
            }
            for qid in test_ids
        }
        idf, idf_manifest = _build_idf(corpus_text.values())
        features = static_features_for_arm(
            query_ids=test_ids,
            baseline_ids=d8,
            arm_ids=entry_ids,
            rank_maps=rank_maps,
            candidate_vectors=candidate_vectors,
            query_vectors=query_vectors,
            corpus_text=corpus_text,
            query_text=query_text,
            idf=idf,
        )
        pairs = [(qid, cid) for qid in test_ids for cid in entry_ids[qid]]
        predictions, prediction_audit = _predict_frozen_model(
            cfg, pairs, features, staging
        )
        historical_rows = _read_jsonl(Path(cfg["historical_system_rankings"]))
        historical = {
            (str(row["query_id"]), str(row["arm"])): list(map(str, row["comment_ids"]))
            for row in historical_rows
            if str(row["backend"]) == "e5"
            and str(row["arm"]) in {"raw_d8", "dense_d12_one_swap"}
        }
        if len(historical) != len(test_ids) * 2:
            raise ValueError("historical T0/T1 Test200 system coverage changed")
        systems, candidate_rows = [], []
        selection_rows = []
        for qid in test_ids:
            if historical[(qid, "raw_d8")] != d8[qid]:
                raise ValueError(f"{qid}: historical baseline is not E5 Dense Top-8")
            model_scores = {cid: predictions[(qid, cid)] for cid in entry_ids[qid]}
            t2 = _top_ids(model_scores, int(cfg["final_k"]))
            residual = _rank_prior_fused_scores(
                entry_ids[qid], entry_scores[qid], model_scores,
                mode="score_cc", entry_weight=float(cfg["residual_beta"]),
                k0=int(cfg["rrf_k0"]), normalization=str(cfg["cc_normalization"]),
            )
            selected = {
                ARMS[0]: d8[qid],
                ARMS[1]: historical[(qid, "dense_d12_one_swap")],
                ARMS[2]: t2,
                ARMS[3]: _top_ids(residual, int(cfg["final_k"])),
            }
            if any(len(ids) != 8 or len(set(ids)) != 8 for ids in selected.values()):
                raise ValueError(f"{qid}: a preregistered arm is not unique Top-8")
            for arm, ids in selected.items():
                systems.append({
                    "query_id": qid, "backend": "e5", "arm": arm,
                    "comment_ids": ids,
                    "inference_used_gold_utility": False,
                    "model_refit_after_test_contact": False,
                })
            selection_rows.append({
                "query_id": qid,
                "t0_dense_top8": selected[ARMS[0]],
                "t1_d12_one_swap_huber": selected[ARMS[1]],
                "t2_m50_lambdamart_direct": selected[ARMS[2]],
                "t3_m50_lambdamart_residual_beta0175": selected[ARMS[3]],
                "t2_t0_overlap": len(set(selected[ARMS[2]]) & set(selected[ARMS[0]])),
                "t3_t2_overlap": len(set(selected[ARMS[3]]) & set(selected[ARMS[2]])),
            })
            dense12_tail = set(d50[qid][8:12])
            graph4 = set(graph50[qid][:4])
            graph_all = set(graph50[qid])
            for cid in sorted(set().union(*map(set, selected.values()))):
                candidate_rows.append({
                    "query_id": qid, "backend": "e5", "candidate_id": cid,
                    "in_d8": cid in set(d8[qid]),
                    "in_dense12_tail": cid in dense12_tail,
                    "in_shared_strict_graph4": cid in graph4,
                    "graph_actionable_for_backend": cid in graph_all,
                    "query_text": query_text[qid],
                    "comment_text": corpus_text[cid],
                    "test_utility_present": False,
                })

        _write_jsonl(staging / "system_rankings.jsonl", systems)
        _write_jsonl(staging / "PRIVATE" / "candidate_union_for_judging.jsonl", candidate_rows)
        _write_jsonl(staging / "PRIVATE" / "selected_sets.jsonl", selection_rows)
        _write_jsonl(staging / "PRIVATE_graph" / "g50_memberships.jsonl", [
            {"query_id": qid, "comment_ids": graph50[qid]}
            for qid in test_ids
        ])
        p3 = {
            "schema": "rq2b-test200-rrf-dense-m50-identity-v1",
            "queries": len(test_ids),
            "candidate_set_identical_queries": sum(row["candidate_set_identical"] for row in p3_rows),
            "candidate_order_identical_queries": sum(row["candidate_order_identical"] for row in p3_rows),
            "universal_entry_identity_migrated": all(row["candidate_set_identical"] for row in p3_rows),
            "claim_rule": (
                "If fewer than 200/200 sets are identical, the Development300 entry-identity "
                "claim is not generalized to Test200; RRF remains the preregistered entry."
            ),
            "rows": p3_rows,
        }
        _write_json(staging / "p3_rrf_dense_m50_identity.json", p3)
        model_record = {
            "schema": "rq2b-test200-frozen-selector-transfer-v1",
            "training_performed_after_test_contact": False,
            "prediction_audit": prediction_audit,
            "residual_beta": float(cfg["residual_beta"]),
            "residual_definition": "Score-CC query-local minmax; beta weights RRF entry",
            "idf_manifest": idf_manifest,
            "p3_identity_summary": {key: p3[key] for key in (
                "queries", "candidate_set_identical_queries",
                "candidate_order_identical_queries", "universal_entry_identity_migrated",
            )},
        }
        _write_json(staging / "selector_transfer_models.json", model_record)
        unique_pairs = {(row["query_id"], row["candidate_id"]) for row in candidate_rows}
        manifest = {
            **contract,
            "phase": "prepare-local-complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "output_version": str(cfg["version"]),
            "test_split_used": True,
            "test_admin_opened": True,
            "test_outcomes_read": False,
            "test_utility_labels_read": False,
            "external_calls": 0,
            "query_count": len(test_ids),
            "systems": len(systems),
            "backends": list(BACKENDS),
            "arms": list(ARMS),
            "candidate_union_ledger_rows": len(candidate_rows),
            "candidate_union_unique_pairs": len(unique_pairs),
            "p1_final_model_frozen_before_test": True,
            "p2_strict_native_g50_built": True,
            "p3_rrf_dense_identity_checked": True,
            "selection_performed": True,
            "selection_uses_test_utility": False,
            "final_model_refit_after_test_contact": False,
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _registry(path: Path) -> dict[tuple[str, str], dict]:
    rows = _read_jsonl(path)
    result = {(str(row["query_id"]), str(row["comment_id"])): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate utility registry key: {path}")
    return result


def analyze_test_rq2b(cfg: dict) -> dict:
    preflight, preflight_manifest, _, _ = _load_verified_judging_preflight(cfg)
    judging = Path(cfg["output_dir"]) / str(cfg["judging_output_subdir"])
    completion_path = judging / "completion_manifest.json"
    if not completion_path.exists():
        raise FileNotFoundError("complete residual utility judgments are required")
    completion = _read_json(completion_path)
    if completion.get("status") != "UTILITY_V2_COMPLETE_READY_FOR_PAIRED_ANALYSIS":
        raise ValueError("judging is not analysis-ready")
    expected_residual = int(preflight_manifest["residual_unique_pairs"])
    if (
        completion.get("residual_valid") != expected_residual
        or completion.get("residual_expected") != expected_residual
        or completion.get("anchors_valid") != 50
    ):
        raise ValueError("judging completion counts changed")
    new_path = judging / "test_utility_registry.jsonl"
    if _sha256(new_path) != completion["test_utility_registry_sha256"]:
        raise ValueError("new residual registry changed after judging")
    old_path = Path(cfg["existing_test_utility_registry"])
    if _sha256(old_path) != preflight_manifest["existing_registry_sha256"]:
        raise ValueError("reused Test utility registry changed after preflight")
    registry = _registry(old_path)
    new_registry = _registry(new_path)
    overlap = set(registry) & set(new_registry)
    if overlap:
        raise ValueError("new residual registry overlaps the reused registry")
    registry.update(new_registry)
    systems = _read_jsonl(Path(cfg["output_dir"]) / "system_rankings.jsonl")
    qids = _read_ids(Path(cfg["test_ids"]))
    query_scores: dict[tuple[str, str], float] = {}
    per_query = []
    selected_keys = set()
    for row in systems:
        qid, arm = str(row["query_id"]), str(row["arm"])
        ids = list(map(str, row["comment_ids"]))
        missing = [cid for cid in ids if (qid, cid) not in registry]
        if missing:
            raise ValueError(f"{qid}/{arm}: selected evidence lacks utility: {missing}")
        evidence = [registry[(qid, cid)] for cid in ids]
        query_scores[(qid, arm)] = statistics.fmean(
            float(item["utility"]) for item in evidence
        )
        selected_keys.update((qid, cid) for cid in ids)
        per_query.append({
            "query_id": qid,
            "arm": arm,
            "utility_at8": query_scores[(qid, arm)],
            **{
                f"mean_{dimension}": statistics.fmean(
                    float(item[f"label_{dimension}"]) for item in evidence
                )
                for dimension in DIMS_V2
            },
        })
    contrasts = (
        ("C1_T2_minus_T0", ARMS[2], ARMS[0], "primary"),
        ("C2_T2_minus_T1", ARMS[2], ARMS[1], "primary"),
        ("C3_T3_minus_T2", ARMS[3], ARMS[2], "secondary"),
    )
    results = []
    seed, draws = int(cfg["bootstrap_seed"]), int(cfg["bootstrap_draws"])
    for name, left, right, tier in contrasts:
        deltas = [query_scores[(qid, left)] - query_scores[(qid, right)] for qid in qids]
        ci = _paired_bootstrap_ci(deltas, seed=seed, draws=draws)
        mean_delta = statistics.fmean(deltas)
        if ci[0] > 0:
            decision = "CONFIRMED_POSITIVE_TEST_SIGNAL"
        elif ci[1] < 0:
            decision = "CONTRADICTED_BY_NEGATIVE_TEST_SIGNAL"
        elif mean_delta > 0:
            decision = (
                "DIRECTIONALLY_CONSISTENT_BUT_UNDERPOWERED_SECONDARY_CONTRAST"
                if name == "C3_T3_minus_T2"
                else "DIRECTIONALLY_CONSISTENT_INCONCLUSIVE"
            )
        else:
            decision = "NOT_CONFIRMED_CI_CROSSES_ZERO_NOT_EQUIVALENCE"
        results.append({
            "contrast": name, "tier": tier, "left_arm": left, "right_arm": right,
            "queries": len(qids), "mean_delta_utility_at8": mean_delta,
            "paired_query_bootstrap_95ci": list(ci),
            "wins": sum(value > 1e-12 for value in deltas),
            "ties": sum(abs(value) <= 1e-12 for value in deltas),
            "losses": sum(value < -1e-12 for value in deltas),
            "decision": decision,
        })
    analysis = Path(cfg["output_dir"]) / "analysis_rq2b_v1"
    if analysis.exists():
        raise FileExistsError(f"versioned analysis already exists: {analysis}")
    analysis.mkdir(parents=True)
    merged_rows = [registry[key] for key in sorted(selected_keys)]
    _write_jsonl(analysis / "selected_output_utility_registry_union.jsonl", merged_rows)
    _write_jsonl(analysis / "per_query_system_metrics.jsonl", per_query)
    _write_json(analysis / "paired_contrasts.json", results)
    report = {
        "schema": "confirmatory-test200-rq2b-paired-analysis-v1",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "query_count": len(qids),
        "bootstrap": {"unit": "query", "draws": draws, "seed": seed},
        "utility_judgment_role": "LLM simulated-user silver; not human gold",
        "test_used_for_tuning": False,
        "comparisons": results,
        "preregistered_interpretation": {
            "C1": "T2 minus T0 is the primary baseline comparison.",
            "C2": "T2 minus T1 is the primary historical-anchor comparison.",
            "C3": (
                "T3 minus T2 is secondary with preregistered approximate power 0.70; "
                "a positive point estimate with a zero-crossing CI is directionally "
                "consistent but underpowered, not evidence that the residual is invalid."
            ),
            "common": "A CI crossing zero is inconclusive and is not equivalence.",
        },
        "registry_reuse": {
            "reused_pairs_available": len(_registry(old_path)),
            "newly_judged_pairs": len(new_registry),
            "selected_output_union_pairs": len(selected_keys),
        },
        "input_sha256": {
            "preflight_manifest": _sha256(preflight / "manifest.json"),
            "judging_completion_manifest": _sha256(completion_path),
            "existing_registry": _sha256(old_path),
            "new_residual_registry": _sha256(new_path),
            "system_rankings": _sha256(Path(cfg["output_dir"]) / "system_rankings.jsonl"),
        },
    }
    _write_json(analysis / "confirmatory_test_report.json", report)
    return report
