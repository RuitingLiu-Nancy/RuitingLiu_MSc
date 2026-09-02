#!/usr/bin/env python3
"""Matched development-only set-aware evidence-selection ablation.

The runner reuses the frozen out-of-fold candidate-utility predictions and
candidate pools from ``selection_action_space_repair``.  It changes only the
selection rule: lambda=0 reproduces additive constrained selection, while
lambda>0 applies a standard greedy MMR relevance--redundancy trade-off under
the same D8-anchored replacement budget.  Hidden community replies are loaded
only after all selected sets have been frozen and are used as a secondary
post-hoc outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import configuration as project_config
    from evaluation import community_reply_auxiliary as community
    from evaluation.judgment_completeness import complete_utility_v2_rows
    from candidate_pool.run_dense_semantic_drift_rescue_audit import _load_embeddings
    from evidence_selection.run_selection_action_space_repair import (
        POOL_DENSE,
        SCORER_HUBER,
        SCORER_MLP,
        read_jsonl,
        sha256,
        utility_at8,
        write_csv,
        write_json,
    )
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import configuration as project_config
    from evaluation import community_reply_auxiliary as community
    from evaluation.judgment_completeness import complete_utility_v2_rows
    from candidate_pool.run_dense_semantic_drift_rescue_audit import _load_embeddings
    from evidence_selection.run_selection_action_space_repair import (
        POOL_DENSE,
        SCORER_HUBER,
        SCORER_MLP,
        read_jsonl,
        sha256,
        utility_at8,
        write_csv,
        write_json,
    )


ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEY = "set_aware_selection_ablation"
EPS = 1e-12


def _setaware_resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _setaware_utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _setaware_git_head() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()


def _candidate_quality(score: float) -> float:
    """Canonical MMR quality mapping for the frozen 1--7 utility scale."""
    return max(0.0, min(1.0, (float(score) - 1.0) / 6.0))


def anchored_mmr_select(
    d8_ids: list[str],
    pool_ids: list[str],
    scores: dict[str, float],
    vectors: dict[str, np.ndarray],
    replacement_budget: int,
    diversity_lambda: float,
    *,
    k: int = 8,
) -> dict[str, Any]:
    """Greedy MMR with at most ``r`` selected items outside raw D8.

    The scoring rule is the repository's canonical MMR convention:
    ``(1-lambda)*quality - lambda*max_similarity_to_selected``.  The only
    adaptation is the D8-relative feasibility constraint.  Deterministic ties
    prefer higher quality, then a baseline item, then lexical candidate ID.
    """
    baseline = list(dict.fromkeys(map(str, d8_ids)))
    pool = list(dict.fromkeys(map(str, pool_ids)))
    baseline_set = set(baseline)
    if len(baseline) != k or not baseline_set <= set(pool):
        raise ValueError("invalid D8/pool relationship")
    if replacement_budget < 0 or not 0.0 <= diversity_lambda < 1.0:
        raise ValueError("invalid replacement budget or diversity lambda")
    missing_scores = [cid for cid in pool if cid not in scores]
    missing_vectors = [cid for cid in pool if cid not in vectors]
    if missing_scores or missing_vectors:
        raise KeyError(
            f"missing scores={len(missing_scores)} vectors={len(missing_vectors)}"
        )

    quality = {cid: _candidate_quality(scores[cid]) for cid in pool}
    selected: list[str] = []
    remaining = set(pool)
    trace: list[dict[str, Any]] = []
    entrant_count = 0
    for step in range(k):
        best_id: str | None = None
        best_value = -math.inf
        best_quality = -math.inf
        best_baseline = False
        best_redundancy = 0.0
        for cid in sorted(remaining):
            is_baseline = cid in baseline_set
            if not is_baseline and entrant_count >= replacement_budget:
                continue
            redundancy = 0.0
            if selected:
                redundancy = max(
                    float(vectors[cid] @ vectors[chosen]) for chosen in selected
                )
            value = (
                (1.0 - diversity_lambda) * quality[cid]
                - diversity_lambda * redundancy
            )
            better = value > best_value + EPS
            tied = abs(value - best_value) <= EPS
            if tied and quality[cid] > best_quality + EPS:
                better = True
            elif tied and abs(quality[cid] - best_quality) <= EPS:
                if is_baseline and not best_baseline:
                    better = True
                elif is_baseline == best_baseline and (
                    best_id is None or cid < best_id
                ):
                    better = True
            if better:
                best_id = cid
                best_value = value
                best_quality = quality[cid]
                best_baseline = is_baseline
                best_redundancy = redundancy
        if best_id is None:
            raise AssertionError(f"no feasible candidate at selection step {step}")
        selected.append(best_id)
        remaining.remove(best_id)
        entrant_count += int(best_id not in baseline_set)
        trace.append({
            "step": step + 1,
            "candidate_id": best_id,
            "predicted_utility": float(scores[best_id]),
            "quality_01": float(quality[best_id]),
            "max_similarity_to_selected": float(best_redundancy),
            "mmr_score": float(best_value),
            "is_d8": bool(best_baseline),
            "entrant_count_after_step": entrant_count,
        })
    if len(selected) != k or len(set(selected)) != k:
        raise AssertionError("MMR did not produce K unique candidates")
    if len(set(selected) - baseline_set) > replacement_budget:
        raise AssertionError("MMR exceeded the replacement budget")
    return {
        "selected_ids": selected,
        "replacement_count": len(set(selected) - baseline_set),
        "trace": trace,
    }


def _mean_pairwise_similarity(
    ids: list[str], vectors: dict[str, np.ndarray]
) -> float:
    values = [
        float(vectors[ids[i]] @ vectors[ids[j]])
        for i in range(len(ids)) for j in range(i + 1, len(ids))
    ]
    return statistics.fmean(values) if values else 0.0


def _alignment(
    ids: list[str],
    query_id: str,
    corpus: dict[str, str],
    references: dict[str, list[dict]],
    embeddings: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, float]:
    result = community.alignment(
        [corpus[cid] for cid in ids],
        [row["text"] for row in references[query_id]],
        embeddings,
        threshold,
    )
    result["bialign_f1_at8"] = community.bidirectional_f(
        result["cra_at8"], result["rcc_at8"]
    )
    return result


def _load_bge_embeddings(
    cfg: dict,
    corpus: dict[str, str],
    references: dict[str, list[dict]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    formal_rows = read_jsonl(cfg["formal_union"])
    texts = [row["text"] for items in references.values() for row in items]
    texts.extend(corpus[str(row["comment_id"])] for row in formal_rows)
    keyed = {community.text_sha(text): text for text in texts}
    ordered = sorted(keyed)
    _, community_cfg = community.load_config()
    encoder = community_cfg["semantic_encoder"]
    cache_payload = {
        "model": encoder["model_id"],
        "revision": encoder["revision"],
        "max_sequence_length": encoder["max_sequence_length"],
        "text_hashes": ordered,
    }
    cache_key = community.sha256_bytes(
        json.dumps(cache_payload, sort_keys=True).encode("utf-8")
    )
    expected_name = f"semantic_embeddings_{cache_key[:16]}.npz"
    if cfg["bge_cache"].name != expected_name:
        raise ValueError(
            f"BGE cache identity changed: {cfg['bge_cache'].name} != {expected_name}"
        )
    matrix = np.load(cfg["bge_cache"], allow_pickle=False)["embeddings"]
    if matrix.shape != (len(ordered), 1024):
        raise ValueError(f"unexpected BGE cache shape {matrix.shape}")
    manifest = json.loads(cfg["bge_manifest"].read_text(encoding="utf-8"))
    recorded = manifest["bge_encoder"]["cache_sha256"]
    if sha256(cfg["bge_cache"]) != recorded:
        raise ValueError("BGE cache hash no longer matches triangulation manifest")
    return dict(zip(ordered, matrix, strict=True)), {
        "path": str(cfg["bge_cache"].relative_to(ROOT)),
        "sha256": recorded,
        "rows": int(matrix.shape[0]),
        "dimension": int(matrix.shape[1]),
        "model_id": encoder["model_id"],
        "revision": encoder["revision"],
        "used_for_selection": False,
    }


def _bootstrap_interval(
    values: list[float], indices: np.ndarray
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    draws = array[indices].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _setaware_group_summary(
    rows: list[dict], qids: list[str], *, include_community: bool
) -> list[dict]:
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (
            row["backend"], row["scorer"], int(row["dense_depth"]),
            int(row["replacement_budget"]), float(row["diversity_lambda"]),
        )
        grouped[key][str(row["query_id"])] = row
    output = []
    for key, by_query in sorted(grouped.items(), key=str):
        if set(by_query) != set(qids):
            raise AssertionError("selection summary lost query coverage")
        values = [by_query[qid] for qid in qids]
        record = {
            "backend": key[0],
            "scorer": key[1],
            "dense_depth": key[2],
            "replacement_budget": key[3],
            "diversity_lambda": key[4],
            "queries": len(values),
            "mean_selected_utility_at8": statistics.fmean(
                float(row["selected_utility_at8"]) for row in values
            ),
            "mean_gain_over_raw_d8": statistics.fmean(
                float(row["gain_over_raw_d8"]) for row in values
            ),
            "harmful_query_rate_vs_raw_d8": statistics.fmean(
                float(row["harmful_vs_raw_d8"]) for row in values
            ),
            "mean_replacement_count": statistics.fmean(
                int(row["replacement_count"]) for row in values
            ),
            "mean_pairwise_backend_similarity": statistics.fmean(
                float(row["mean_pairwise_backend_similarity"]) for row in values
            ),
        }
        if include_community:
            record.update({
                "mean_cra_at8": statistics.fmean(
                    float(row["cra_at8"]) for row in values
                ),
                "mean_rcc_at8": statistics.fmean(
                    float(row["rcc_at8"]) for row in values
                ),
                "mean_bialign_f1_at8": statistics.fmean(
                    float(row["bialign_f1_at8"]) for row in values
                ),
            })
        output.append(record)
    return output


def _setaware_paired_contrasts(
    rows: list[dict],
    qids: list[str],
    lambdas: list[float],
    indices: np.ndarray,
    *,
    include_community: bool,
) -> list[dict]:
    lookup = {
        (
            row["backend"], row["scorer"], int(row["dense_depth"]),
            int(row["replacement_budget"]), float(row["diversity_lambda"]),
            str(row["query_id"]),
        ): row
        for row in rows
    }
    cells = sorted({key[:4] for key in lookup}, key=str)
    output = []
    fields = {
        "utility": "selected_utility_at8",
        "pairwise_similarity": "mean_pairwise_backend_similarity",
    }
    if include_community:
        fields.update({
            "cra": "cra_at8",
            "rcc": "rcc_at8",
            "bialign_f1": "bialign_f1_at8",
        })
    for cell in cells:
        backend, scorer, depth, budget = cell
        additive = [lookup[(*cell, 0.0, qid)] for qid in qids]
        for lam in lambdas:
            if abs(lam) <= EPS:
                continue
            aware = [lookup[(*cell, lam, qid)] for qid in qids]
            record = {
                "backend": backend,
                "scorer": scorer,
                "dense_depth": depth,
                "replacement_budget": budget,
                "diversity_lambda": lam,
                "queries": len(qids),
                "changed_query_count": sum(
                    set(a["selected_comment_ids"]) != set(b["selected_comment_ids"])
                    for a, b in zip(aware, additive, strict=True)
                ),
            }
            record["changed_query_fraction"] = record["changed_query_count"] / len(qids)
            utility_deltas = [
                float(a["selected_utility_at8"]) - float(b["selected_utility_at8"])
                for a, b in zip(aware, additive, strict=True)
            ]
            record["utility_win_count"] = sum(value > EPS for value in utility_deltas)
            record["utility_tie_count"] = sum(abs(value) <= EPS for value in utility_deltas)
            record["utility_loss_count"] = sum(value < -EPS for value in utility_deltas)
            for label, field in fields.items():
                deltas = [
                    float(a[field]) - float(b[field])
                    for a, b in zip(aware, additive, strict=True)
                ]
                mean, lo, hi = _bootstrap_interval(deltas, indices)
                record[f"mean_delta_{label}"] = mean
                record[f"delta_{label}_95ci_lo"] = lo
                record[f"delta_{label}_95ci_hi"] = hi
            record["harmful_rate_delta_vs_additive"] = statistics.fmean(
                float(a["harmful_vs_raw_d8"]) - float(b["harmful_vs_raw_d8"])
                for a, b in zip(aware, additive, strict=True)
            )
            output.append(record)
    return output


def _format_ci(row: dict, label: str, digits: int = 4) -> str:
    return (
        f"{float(row[f'mean_delta_{label}']):+.{digits}f} "
        f"[{float(row[f'delta_{label}_95ci_lo']):+.{digits}f}, "
        f"{float(row[f'delta_{label}_95ci_hi']):+.{digits}f}]"
    )


def _setaware_write_result_documents(
    out: Path,
    contrasts: list[dict],
    cfg: dict,
    outcomes: dict[str, Any],
) -> None:
    include_community = bool(cfg.get("evaluate_community_correspondence", True))
    primary_key = cfg["primary_cell"]
    primary = next(
        row for row in contrasts
        if row["backend"] == primary_key["backend"]
        and row["scorer"] == primary_key["scorer"]
        and int(row["dense_depth"]) == int(primary_key["dense_depth"])
        and int(row["replacement_budget"]) == int(primary_key["replacement_budget"])
        and math.isclose(
            float(row["diversity_lambda"]),
            float(cfg["primary_diversity_lambda"]),
        )
    )
    lines = [
        "# Matched set-aware evidence-selection ablation",
        "",
        "## Frozen primary result",
        "",
        f"- Set-aware minus additive Utility@8: `{_format_ci(primary, 'utility')}`.",
        *(
            [
                f"- Set-aware minus additive BiAlignF1: `{_format_ci(primary, 'bialign_f1', 6)}`.",
                f"- Set-aware minus additive RCC: `{_format_ci(primary, 'rcc', 6)}`.",
            ]
            if include_community
            else [
                "- Community correspondence was not re-evaluated in this frozen run because the expanded reference package had not yet been materialised; later Development300 candidate-level C is a separate post-hoc output."
            ]
        ),
        f"- Mean pairwise backend similarity change: `{_format_ci(primary, 'pairwise_similarity', 4)}`.",
        f"- Changed sets: `{primary['changed_query_count']}/{primary['queries']}` queries; "
        f"U win/tie/loss `{primary['utility_win_count']}/{primary['utility_tie_count']}/{primary['utility_loss_count']}`.",
        "",
        "## Pre-registered verdicts",
        "",
        f"- Primary utility gain: **{outcomes['primary_utility_gain']}**.",
        f"- Robust M12/r2 gain: **{outcomes['robust_m12_r2_gain']}**.",
        f"- Ecological convergence: **{outcomes['ecological_convergence']}**.",
        "",
        "Lower semantic redundancy is a mechanical property of the selector, not by itself an effectiveness result. Community correspondence remained post-hoc and did not tune any policy.",
    ]
    (out / "RESULTS_INTERPRETATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    insertion = [
        "# Chapter 5 insertion boundary",
        "",
        "This experiment belongs in the reranking/selection-capacity subsection, after the additive replacement frontier and before fresh-policy validation. It must not be placed in the Graph-contribution subsection.",
        "",
        "The accurate method label is **replacement-budget-aware greedy MMR over frozen OOF candidate-utility scores**. It is a lightweight set-aware adaptation, not a reproduction of JPR or SetR and not a new neural reranker architecture.",
        "",
        (
            f"Primary result: set-aware minus additive Utility@8 `{_format_ci(primary, 'utility')}`; BiAlignF1 `{_format_ci(primary, 'bialign_f1', 6)}`."
            if include_community else
            f"Primary result: set-aware minus additive Utility@8 `{_format_ci(primary, 'utility')}`; C was not re-evaluated on the expanded cohort."
        ),
        "",
        "The contribution claim may be strengthened only if the corresponding pre-registered verdict is supported; otherwise the experiment is a bounded negative or qualifying ablation showing whether explicit composition adds value beyond replacement capacity.",
    ]
    (out / "THESIS_INSERTION.md").write_text(
        "\n".join(insertion) + "\n", encoding="utf-8"
    )

    table_rows = [
        row for row in contrasts
        if math.isclose(
            float(row["diversity_lambda"]),
            float(cfg["primary_diversity_lambda"]),
        ) and (
            (int(row["dense_depth"]) == 12 and int(row["replacement_budget"]) == 2)
            or (int(row["dense_depth"]) == 50 and int(row["replacement_budget"]) == 8)
        )
    ]
    tex = [
        r"\begin{tabular}{lllrrr}",
        r"\hline",
        r"Backend & Scorer & Cell & $\Delta U@8$ (95\% CI) & $\Delta C_{F1}$ & Changed \\",
        r"\hline",
    ]
    for row in table_rows:
        scorer = "Huber" if row["scorer"] == SCORER_HUBER else "Small MLP"
        tex.append(
            f"{row['backend'].upper()} & {scorer} & M{row['dense_depth']}, r={row['replacement_budget']} & "
            f"{_format_ci(row, 'utility', 3)} & "
            f"{(_format_ci(row, 'bialign_f1', 5) if include_community else 'N/A')} & "
            f"{row['changed_query_count']}/{row['queries']} \\\\"
        )
    tex.extend([r"\hline", r"\end{tabular}"])
    (out / "table_set_aware_ablation.tex").write_text(
        "\n".join(tex) + "\n", encoding="utf-8"
    )


def _setaware_resolve_config(config_key: str) -> dict[str, Any]:
    raw = dict(project_config.load()[config_key])
    path_keys = [
        "output_dir", "preregistration", "source_repair_dir", "utility_registry",
        "queries", "corpus",
    ]
    if bool(raw.get("evaluate_community_correspondence", True)):
        path_keys.extend([
            "formal_union", "community_reference_dir", "bge_cache", "bge_manifest"
        ])
    for key in path_keys:
        raw[key] = _setaware_resolve(raw[key])
    raw["embeddings"] = {
        backend: {name: _setaware_resolve(value) for name, value in paths.items()}
        for backend, paths in raw["embeddings"].items()
    }
    if bool(raw["allow_external_calls"]) or bool(raw["allow_frozen_test"]):
        raise ValueError("set-aware ablation must remain local development-only")
    if list(raw["backends"]) != ["minilm", "e5"]:
        raise ValueError("backend grid changed")
    if list(raw["scorers"]) != [SCORER_HUBER, SCORER_MLP]:
        raise ValueError("scorer grid changed")
    if list(map(int, raw["dense_depths"])) != [12, 50]:
        raise ValueError("depth grid changed")
    if list(map(int, raw["replacement_budgets"])) != [1, 2, 4, 8]:
        raise ValueError("replacement-budget grid changed")
    if list(map(float, raw["diversity_lambdas"])) != [0.0, 0.1, 0.25]:
        raise ValueError("diversity grid changed")
    return raw


def run(output_dir: Path | None = None, config_key: str = CONFIG_KEY) -> dict:
    started = time.perf_counter()
    cfg = _setaware_resolve_config(config_key)
    destination = (output_dir or cfg["output_dir"]).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")

    source = cfg["source_repair_dir"]
    required_source = {
        "config": source / "config.json",
        "manifest": source / "reproduction_manifest.json",
        "pool": source / "candidate_pool_manifest.parquet",
        "predictions": source / "oof_candidate_predictions.parquet",
        "sets": source / "selected_sets.parquet",
        "qids": source / "query_ids.txt",
    }
    include_community = bool(cfg.get("evaluate_community_correspondence", True))
    required_paths = [*required_source.values(), cfg["preregistration"]]
    if include_community:
        required_paths.append(cfg["bge_cache"])
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    source_manifest = json.loads(required_source["manifest"].read_text(encoding="utf-8"))
    if source_manifest.get("status") != "COMPLETE" or source_manifest["invariants"]["frozen_test_read"]:
        raise ValueError("source action-space package is not a valid frozen development package")

    qids = [
        line.strip() for line in required_source["qids"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_queries = int(cfg.get("expected_query_count", 100))
    if len(qids) != expected_queries or len(set(qids)) != expected_queries:
        raise ValueError("development query identity changed")
    qid_set = set(qids)

    pool_frame = pd.read_parquet(required_source["pool"])
    prediction_frame = pd.read_parquet(required_source["predictions"])
    source_sets_frame = pd.read_parquet(required_source["sets"])
    pool_frame = pool_frame[
        (pool_frame["pool_family"] == POOL_DENSE)
        & pool_frame["dense_depth"].isin(list(map(int, cfg["dense_depths"])))
    ].copy()
    prediction_frame = prediction_frame[
        prediction_frame["backend"].isin(cfg["backends"])
        & prediction_frame["scorer"].isin(cfg["scorers"])
    ].copy()
    source_sets_frame = source_sets_frame[
        (source_sets_frame["selection_kind"] == "learned")
        & (source_sets_frame["pool_family"] == POOL_DENSE)
        & source_sets_frame["dense_depth"].isin(list(map(int, cfg["dense_depths"])))
        & source_sets_frame["scorer"].isin(cfg["scorers"])
        & source_sets_frame["replacement_budget"].isin(
            list(map(int, cfg["replacement_budgets"]))
        )
    ].copy()

    pool_lookup: dict[tuple[str, int, str], list[str]] = {}
    d8_lookup: dict[tuple[str, int, str], list[str]] = {}
    for key, rows in pool_frame.groupby(["backend", "dense_depth", "query_id"]):
        backend, depth, qid = str(key[0]), int(key[1]), str(key[2])
        ordered = rows.sort_values("pool_position")
        ids = list(map(str, ordered["candidate_id"]))
        if len(ids) != depth or len(set(ids)) != depth:
            raise AssertionError(f"{backend}/M{depth}/{qid}: Dense pool drift")
        d8 = list(map(str, ordered.loc[ordered["in_d8"], "candidate_id"]))
        if len(d8) != 8:
            raise AssertionError("D8 membership drift")
        pool_lookup[(backend, depth, qid)] = ids
        d8_lookup[(backend, depth, qid)] = d8
    expected_pool_keys = {
        (backend, depth, qid)
        for backend in cfg["backends"]
        for depth in map(int, cfg["dense_depths"])
        for qid in qids
    }
    if set(pool_lookup) != expected_pool_keys:
        raise AssertionError("pool grid is incomplete")

    predictions = {
        (str(row.backend), str(row.scorer), str(row.query_id), str(row.candidate_id)):
        float(row.oof_prediction_mean)
        for row in prediction_frame.itertuples(index=False)
    }
    source_sets = {
        (
            str(row.backend), str(row.scorer), int(row.dense_depth),
            int(row.replacement_budget), str(row.query_id),
        ): list(map(str, row.selected_comment_ids))
        for row in source_sets_frame.itertuples(index=False)
    }

    complete_rows, registry = complete_utility_v2_rows(read_jsonl(cfg["utility_registry"]))
    expected_registry_rows = int(cfg.get("expected_complete_registry_rows", 19_813))
    if len(complete_rows) != expected_registry_rows:
        raise ValueError("coverage-complete utility registry changed")

    corpus_rows = json.loads(cfg["corpus"].read_text(encoding="utf-8"))
    corpus = {str(row["title"]): str(row["text"]) for row in corpus_rows}
    if len(corpus) != 19013:
        raise ValueError("fixed corpus identity changed")
    references: dict[str, list[dict]] = {}
    bge_embeddings: dict[str, np.ndarray] = {}
    if include_community:
        references = community.load_admin_references(cfg["community_reference_dir"])
        expected_hidden_replies = int(cfg.get("expected_hidden_replies", 1_749))
        if (
            set(references) != qid_set
            or sum(map(len, references.values())) != expected_hidden_replies
        ):
            raise ValueError("development community-reference identity changed")
        bge_embeddings, bge_audit = _load_bge_embeddings(cfg, corpus, references)
    else:
        bge_audit = {
            "evaluated": False,
            "reason": "no frozen community-reference package for expanded cohort",
            "used_for_selection": False,
        }

    backend_vectors: dict[str, dict[str, np.ndarray]] = {}
    embedding_audit = {}
    for backend in cfg["backends"]:
        vectors, _, corpus_text, _, matrix, ids = _load_embeddings(
            corpus_path=cfg["corpus"],
            corpus_embeddings_path=cfg["embeddings"][backend]["corpus"],
            queries_path=cfg["queries"],
            query_embeddings_path=cfg["embeddings"][backend]["query"],
            qids=qid_set,
        )
        if corpus_text != corpus:
            raise ValueError("embedding corpus text identity changed")
        required_ids = {
            cid for (pool_backend, _, _), pool in pool_lookup.items()
            if pool_backend == backend for cid in pool
        }
        norms = [float(np.linalg.norm(vectors[cid])) for cid in required_ids]
        if min(norms) < 0.999 or max(norms) > 1.001:
            raise ValueError("backend-local candidate vectors are not normalized")
        backend_vectors[backend] = vectors
        embedding_audit[backend] = {
            "corpus_path": str(cfg["embeddings"][backend]["corpus"].relative_to(ROOT)),
            "corpus_sha256": sha256(cfg["embeddings"][backend]["corpus"]),
            "query_path": str(cfg["embeddings"][backend]["query"].relative_to(ROOT)),
            "query_sha256": sha256(cfg["embeddings"][backend]["query"]),
            "corpus_rows": len(ids),
            "dimension": int(matrix.shape[1]),
        }

    alignment_cache: dict[tuple[str, tuple[str, ...]], dict[str, float]] = {}
    selection_rows: list[dict[str, Any]] = []
    lambda_zero_checks = 0
    for backend in cfg["backends"]:
        vectors = backend_vectors[backend]
        for scorer in cfg["scorers"]:
            for depth in map(int, cfg["dense_depths"]):
                for budget in map(int, cfg["replacement_budgets"]):
                    for qid in qids:
                        pool = pool_lookup[(backend, depth, qid)]
                        d8 = d8_lookup[(backend, depth, qid)]
                        score_map = {
                            cid: predictions[(backend, scorer, qid, cid)] for cid in pool
                        }
                        raw_utility = utility_at8(d8, qid, registry)
                        additive_ids = source_sets[(backend, scorer, depth, budget, qid)]
                        for diversity_lambda in map(float, cfg["diversity_lambdas"]):
                            result = anchored_mmr_select(
                                d8, pool, score_map, vectors, budget,
                                diversity_lambda, k=int(cfg["final_k"]),
                            )
                            selected = list(result["selected_ids"])
                            if abs(diversity_lambda) <= EPS:
                                lambda_zero_checks += 1
                                if set(selected) != set(additive_ids):
                                    raise AssertionError(
                                        f"lambda=0 failed additive reproduction: "
                                        f"{backend}/{scorer}/M{depth}/r{budget}/{qid}"
                                    )
                            selected_utility = utility_at8(selected, qid, registry)
                            aligned: dict[str, float] = {}
                            if include_community:
                                alignment_key = (qid, tuple(sorted(selected)))
                                if alignment_key not in alignment_cache:
                                    alignment_cache[alignment_key] = _alignment(
                                        selected, qid, corpus, references, bge_embeddings,
                                        float(cfg["correspondence_threshold"]),
                                    )
                                aligned = alignment_cache[alignment_key]
                            selection_rows.append({
                                "backend": backend,
                                "scorer": scorer,
                                "pool_family": POOL_DENSE,
                                "dense_depth": depth,
                                "replacement_budget": budget,
                                "diversity_lambda": diversity_lambda,
                                "query_id": qid,
                                "selected_comment_ids": selected,
                                "selected_utility_at8": selected_utility,
                                "raw_d8_utility_at8": raw_utility,
                                "gain_over_raw_d8": selected_utility - raw_utility,
                                "harmful_vs_raw_d8": selected_utility < raw_utility - EPS,
                                "replacement_count": int(result["replacement_count"]),
                                "mean_pairwise_backend_similarity": _mean_pairwise_similarity(
                                    selected, vectors
                                ),
                                "selection_trace_json": json.dumps(
                                    result["trace"], ensure_ascii=False, sort_keys=True
                                ),
                                "candidate_quality_target": "frozen OOF predicted utility-v2",
                                "community_used_in_selection": False,
                                "current_query_utility_used_in_selection": False,
                                "route_identity_used_in_selection": False,
                                **aligned,
                            })
    expected_lambda_zero_checks = (
        len(cfg["backends"]) * len(cfg["scorers"])
        * len(cfg["dense_depths"]) * len(cfg["replacement_budgets"]) * len(qids)
    )
    if lambda_zero_checks != expected_lambda_zero_checks:
        raise AssertionError("lambda=0 reproduction coverage mismatch")

    rng = np.random.default_rng(int(cfg["bootstrap_seed"]))
    bootstrap_indices = rng.integers(
        0, len(qids), size=(int(cfg["bootstrap_samples"]), len(qids))
    )
    summary_rows = _setaware_group_summary(
        selection_rows, qids, include_community=include_community
    )
    contrast_rows = _setaware_paired_contrasts(
        selection_rows, qids, list(map(float, cfg["diversity_lambdas"])),
        bootstrap_indices, include_community=include_community,
    )

    primary_key = cfg["primary_cell"]
    primary = next(
        row for row in contrast_rows
        if row["backend"] == primary_key["backend"]
        and row["scorer"] == primary_key["scorer"]
        and int(row["dense_depth"]) == int(primary_key["dense_depth"])
        and int(row["replacement_budget"]) == int(primary_key["replacement_budget"])
        and math.isclose(
            float(row["diversity_lambda"]),
            float(cfg["primary_diversity_lambda"]),
        )
    )
    m12_r2 = [
        row for row in contrast_rows
        if int(row["dense_depth"]) == 12
        and int(row["replacement_budget"]) == 2
        and math.isclose(
            float(row["diversity_lambda"]),
            float(cfg["primary_diversity_lambda"]),
        )
    ]
    if len(m12_r2) != 4:
        raise AssertionError("M12/r2 robustness grid is incomplete")
    outcomes = {
        "primary_utility_gain": (
            "SUPPORTED" if float(primary["delta_utility_95ci_lo"]) > 0
            else "NOT_SUPPORTED"
        ),
        "robust_m12_r2_gain": (
            "SUPPORTED" if all(float(row["delta_utility_95ci_lo"]) > 0 for row in m12_r2)
            else "NOT_SUPPORTED"
        ),
        "ecological_convergence": (
            "SUPPORTED" if include_community
            and float(primary["delta_bialign_f1_95ci_lo"]) > 0
            else ("NOT_SUPPORTED" if include_community else "NOT_EVALUATED")
        ),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary:
        out = Path(temporary)
        pq.write_table(
            pa.Table.from_pylist(selection_rows),
            out / "selected_sets.parquet", compression="zstd",
        )
        write_csv(out / "selection_summary.csv", summary_rows)
        write_csv(out / "paired_contrasts.csv", contrast_rows)
        config_payload = {
            "schema": "set-aware-selection-ablation-config-v1",
            "version": cfg["version"],
            "scope": cfg.get("scope_label", "fully judged development100 Dense-only"),
            "external_calls": 0,
            "frozen_test_read": False,
            "backends": list(cfg["backends"]),
            "scorers": list(cfg["scorers"]),
            "dense_depths": list(map(int, cfg["dense_depths"])),
            "replacement_budgets": list(map(int, cfg["replacement_budgets"])),
            "diversity_lambdas": list(map(float, cfg["diversity_lambdas"])),
            "primary_diversity_lambda": float(cfg["primary_diversity_lambda"]),
            "primary_cell": dict(cfg["primary_cell"]),
            "selection_rule": "replacement-budget-aware greedy MMR",
            "quality_mapping": "clip((OOF predicted utility - 1) / 6, 0, 1)",
            "redundancy": "maximum backend-local candidate cosine to already selected items",
            "tie_break": "higher quality, then D8 membership, then candidate_id",
            "community_role": (
                "post-hoc only; never used for selection or tuning"
                if include_community else
                "not re-evaluated on this cohort; Development100 result retained"
            ),
            "success_rules": dict(cfg["success_rules"]),
            "bootstrap_samples": int(cfg["bootstrap_samples"]),
            "bootstrap_seed": int(cfg["bootstrap_seed"]),
            "bootstrap_indices_sha256": _object_sha256(bootstrap_indices.tolist()),
            "embedding_inputs": embedding_audit,
            "bge_correspondence_input": bge_audit,
            "preregistration": {
                "path": str(cfg["preregistration"].relative_to(ROOT)),
                "sha256": sha256(cfg["preregistration"]),
            },
        }
        write_json(out / "config.json", config_payload)
        _setaware_write_result_documents(out, contrast_rows, cfg, outcomes)

        expected = {
            "config.json", "selected_sets.parquet", "selection_summary.csv",
            "paired_contrasts.csv", "table_set_aware_ablation.tex",
            "RESULTS_INTERPRETATION.md", "THESIS_INSERTION.md",
        }
        if {path.name for path in out.iterdir()} != expected:
            raise AssertionError("output package incomplete before manifest")
        manifest = {
            "schema": "set-aware-selection-ablation-reproduction-v1",
            "status": "COMPLETE",
            "created_utc": _setaware_utc_now(),
            "runtime_seconds": time.perf_counter() - started,
            "command": "PYTHONPATH=.. .venv-reranker-repro/bin/python evidence_selection/run_set_aware_selection_ablation.py",
            "git_head": _setaware_git_head(),
            "implementation": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "development_queries": len(qids),
            "selected_set_rows": len(selection_rows),
            "summary_rows": len(summary_rows),
            "paired_contrast_rows": len(contrast_rows),
            "unique_set_correspondence_evaluations": len(alignment_cache),
            "lambda_zero_additive_reproduction_checks": lambda_zero_checks,
            "outcomes": outcomes,
            "invariants": {
                "external_calls": 0,
                "frozen_test_read": False,
                "graph_candidates_in_primary_analysis": False,
                "same_oof_predictions_as_action_space_repair": True,
                "same_dense_pools_as_action_space_repair": True,
                "same_replacement_budgets_as_action_space_repair": True,
                "lambda_zero_exactly_reproduces_additive_sets": True,
                "current_query_utility_used_in_selection": False,
                "community_replies_used_in_selection_or_tuning": False,
                "community_correspondence_evaluated": include_community,
                "explicit_route_identity_used_in_selection": False,
                "all_sets_size_8_unique": all(
                    len(row["selected_comment_ids"]) == 8
                    and len(set(row["selected_comment_ids"])) == 8
                    for row in selection_rows
                ),
                "all_replacement_budgets_respected": all(
                    int(row["replacement_count"]) <= int(row["replacement_budget"])
                    for row in selection_rows
                ),
            },
            "input_hashes": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in [
                    *required_source.values(), cfg["utility_registry"], cfg["queries"],
                    cfg["corpus"], cfg["preregistration"],
                    *(
                        [cfg["formal_union"], cfg["bge_cache"], cfg["bge_manifest"]]
                        if include_community else []
                    ),
                ]
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
            },
            "output_hashes": {
                path.name: sha256(path) for path in sorted(out.iterdir())
            },
        }
        write_json(out / "reproduction_manifest.json", manifest)
        expected.add("reproduction_manifest.json")
        if {path.name for path in out.iterdir()} != expected:
            raise AssertionError("final output package is incomplete")
        os.replace(out, destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-key", default=CONFIG_KEY)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run(args.output_dir, args.config_key)
    print(json.dumps({
        "status": result["status"],
        "output": str((args.output_dir or _setaware_resolve_config(args.config_key)["output_dir"]).resolve()),
        "runtime_seconds": result["runtime_seconds"],
        "outcomes": result["outcomes"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
