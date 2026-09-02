#!/usr/bin/env python3
"""Post-hoc set-level community correspondence for frozen RQ2b outputs.

This audit does not train, tune, select, encode, or call an external service.
It reads the already frozen Development100 action-space and set-aware packages,
then opens the hidden-reply package and applies the canonical CRA/RCC/
BiAlignF1/BestAlign implementation.  Raw Dense-8 correspondence is the only
new per-set metric evaluation; all non-raw metrics are verified against the
frozen set-aware package before paired reporting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow
import scipy
from scipy.stats import spearmanr

try:
    import configuration as project_config
    from evaluation import community_reply_auxiliary as community
    from evaluation.judgment_completeness import complete_utility_v2_rows
    from evidence_selection import run_set_aware_selection_ablation as setaware
    from evidence_selection.run_selection_action_space_repair import read_jsonl, sha256
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import configuration as project_config
    from evaluation import community_reply_auxiliary as community
    from evaluation.judgment_completeness import complete_utility_v2_rows
    from evidence_selection import run_set_aware_selection_ablation as setaware
    from evidence_selection.run_selection_action_space_repair import read_jsonl, sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEY = "rq2b_set_correspondence"
METHOD_ORDER = ("raw_dense8", "one_swap", "two_swap", "direct", "mmr_r2")
METHOD_LABELS = {
    "raw_dense8": "Raw Dense-8",
    "one_swap": "One-swap",
    "two_swap": "Two-swap",
    "direct": "Direct",
    "mmr_r2": r"MMR $r=2$",
}
METRICS = (
    "utility_at8",
    "cra_at8",
    "rcc_at8",
    "bialign_f1_at8",
    "best_align_at8",
)
METRIC_SHORT = {
    "utility_at8": "u",
    "cra_at8": "cra",
    "rcc_at8": "rcc",
    "bialign_f1_at8": "bialign_f1",
    "best_align_at8": "best_align",
}
CONTRASTS = (
    ("one_swap", "raw_dense8"),
    ("two_swap", "raw_dense8"),
    ("direct", "raw_dense8"),
    ("two_swap", "one_swap"),
    ("direct", "two_swap"),
    ("mmr_r2", "two_swap"),
)
QUADRANT_CONTRASTS = (
    ("one_swap", "raw_dense8"),
    ("two_swap", "raw_dense8"),
    ("direct", "raw_dense8"),
    ("mmr_r2", "raw_dense8"),
    ("two_swap", "one_swap"),
    ("mmr_r2", "two_swap"),
)


def _resolve(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else ROOT / path


def _load_config() -> dict[str, Any]:
    cfg = dict(project_config.load()[CONFIG_KEY])
    for key in ("output_dir", "action_space_dir", "set_aware_dir",
                "community_reference_dir", "corpus_source_map"):
        cfg[key] = _resolve(cfg[key])
    if cfg.get("allow_external_calls") or cfg.get("allow_frozen_test"):
        raise ValueError("RQ2b post-hoc audit must remain local development-only")
    if int(cfg["expected_query_count"]) != 100 or int(cfg["final_k"]) != 8:
        raise ValueError("frozen RQ2b scope changed")
    return cfg


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _object_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite metric: {value}")
    return number


def _load_source_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cid = community.normalize_reddit_id(row["comment_id"])
            post_id = community.normalize_reddit_id(row["post_id"])
            if cid in mapping and mapping[cid] != post_id:
                raise ValueError(f"comment maps to multiple posts: {cid}")
            mapping[cid] = post_id
    return mapping


def _index_rows(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        key = (str(row["method"]), str(row["query_id"]))
        if key in result:
            raise ValueError(f"duplicate method/query row: {key}")
        result[key] = row
    return result


def _mean_ci(values: list[float], indices: np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    draws = array[indices].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _wtl(values: Iterable[float], tolerance: float) -> tuple[int, int, int]:
    vector = np.asarray(list(values), dtype=float)
    return (
        int((vector > tolerance).sum()),
        int((np.abs(vector) <= tolerance).sum()),
        int((vector < -tolerance).sum()),
    )


def _rho(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    result = spearmanr(x, y).statistic
    return float(result) if math.isfinite(float(result)) else None


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}".replace("+0.", "+.").replace("-0.", "-.")


def _ci_cell(row: dict[str, Any], prefix: str, digits: int = 4) -> str:
    return (
        f"{_fmt(float(row[f'delta_{prefix}']), digits)} "
        f"[{_fmt(float(row[f'delta_{prefix}_ci_low']), digits)},"
        f"{_fmt(float(row[f'delta_{prefix}_ci_high']), digits)}]"
    )


def _method_label(method: str) -> str:
    return METHOD_LABELS[method]


def _contrast_label(left: str, right: str) -> str:
    return f"{_method_label(left)} $-$ {_method_label(right)}"


def _build_latex(contrasts: list[dict[str, Any]], quadrants: list[dict[str, Any]]) -> str:
    lookup = {(row["method"], row["baseline"]): row for row in contrasts}
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\small",
        r"\caption{Set-level utility and community-response correspondence for RQ2b selection strategies on the Development100 nested subset. Estimates are paired against the declared baseline, and confidence intervals resample whole queries.}",
        r"\label{tab:ch5_rq2b_set_correspondence}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Method & Baseline & $\Delta U@8$ (95\% CI) & $\Delta$RCC (95\% CI) & $\Delta$BiAlignF1 (95\% CI) \\",
        r"\hline",
    ]
    for left, right in CONTRASTS:
        row = lookup[(left, right)]
        lines.append(
            f"{_method_label(left)} & {_method_label(right)} & "
            f"{_ci_cell(row, 'u')} & {_ci_cell(row, 'rcc', 5)} & "
            f"{_ci_cell(row, 'bialign_f1', 5)} \\\\" 
        )
    lines.extend([
        r"\hline",
        r"\end{tabular}}",
        r"\par\footnotesize The archived MMR arm shares the candidate pool, OOF scores, $K$, and $r$ with additive Two-swap. Its configured greedy tie rule differs from the additive global-set tie rule, so this row is interpreted as a bounded set-objective ablation.",
        r"\end{table}",
        "",
        r"\begin{table}[!ht]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Secondary correspondence diagnostics and paired win--tie--loss counts for the RQ2b Development100 analysis.}",
        r"\label{tab:app_rq2b_set_correspondence_full}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        r"Method & Baseline & $\Delta$CRA (95\% CI) & $\Delta$BestAlign (95\% CI) & U W/T/L & BiAlignF1 W/T/L & Changed sets \\",
        r"\hline",
    ])
    for left, right in CONTRASTS:
        row = lookup[(left, right)]
        lines.append(
            f"{_method_label(left)} & {_method_label(right)} & "
            f"{_ci_cell(row, 'cra', 5)} & {_ci_cell(row, 'best_align', 5)} & "
            f"{row['u_wins']}/{row['u_ties']}/{row['u_losses']} & "
            f"{row['bialign_f1_wins']}/{row['bialign_f1_ties']}/{row['bialign_f1_losses']} & "
            f"{row['changed_set_count']}/100 \\\\"
        )
    lines.extend([
        r"\hline",
        r"\end{tabular}}",
        r"\end{table}",
        "",
        r"\begin{table}[!ht]",
        r"\centering",
        r"\small",
        r"\caption{Joint query-level changes in Utility@8 and BiAlignF1 on changed evidence sets. Unchanged sets are recorded separately.}",
        r"\label{tab:ch5_rq2b_set_quadrants}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Contrast & $U\uparrow,C\uparrow$ & $U\uparrow,C\downarrow$ & $U\downarrow,C\uparrow$ & $U\downarrow,C\downarrow$ & Unchanged \\",
        r"\hline",
    ])
    for row in quadrants:
        lines.append(
            f"{_contrast_label(str(row['method']), str(row['baseline']))} & "
            f"{row['u_up_c_up_count']} & {row['u_up_c_nonpositive_count']} & "
            f"{row['u_nonpositive_c_up_count']} & {row['u_nonpositive_c_nonpositive_count']} & "
            f"{row['unchanged_set_count']} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}}", r"\end{table}", ""])
    return "\n".join(lines)


def _build_patch(
    contrasts: list[dict[str, Any]], quadrants: list[dict[str, Any]]
) -> str:
    lookup = {(row["method"], row["baseline"]): row for row in contrasts}
    quadrant_lookup = {(row["method"], row["baseline"]): row for row in quadrants}
    r1 = lookup[("one_swap", "raw_dense8")]
    r2 = lookup[("two_swap", "raw_dense8")]
    direct = lookup[("direct", "raw_dense8")]
    r2r1 = lookup[("two_swap", "one_swap")]
    dr2 = lookup[("direct", "two_swap")]
    mmr = lookup[("mmr_r2", "two_swap")]
    r2r1_joint = quadrant_lookup[("two_swap", "one_swap")]
    mmr_joint = quadrant_lookup[("mmr_r2", "two_swap")]
    return "\n".join([
        r"\subsubsection{Set-Level Utility and Community-Response Correspondence}",
        r"\label{sec:ch5_rq2b_set_correspondence}",
        "",
        r"The Development100 nested subset supplies 1,749 naturally occurring top-level replies for an independent post-hoc evaluation of the frozen E5 Small-MLP, Dense-$12$, $K=8$ selection outputs. Candidate--Reply Alignment (CRA), Reply--Candidate Coverage (RCC), BiAlignF1, and BestAlign use the frozen BGE-M3 representation defined in Chapter~3. This analysis opens the reply package after loading and validating the frozen evidence sets. The replies contribute exclusively to evaluation; scorer fitting, replacement decisions, and method selection remain fixed from the archived utility-based pipeline.",
        "",
        r"Table~\ref{tab:ch5_rq2b_set_correspondence} compares Raw Dense-8, One-swap, Two-swap, unrestricted Direct selection, and the matched $r=2$, $\lambda=.10$ MMR ablation. Direct is represented once because the Dense-$12$ pool contains four candidates outside $D_8$, making $r=4$ and $r=8$ identical for every query.",
        "",
        r"\input{rq2b_c_table}",
        "",
        f"One-swap increased Utility@8 by {_fmt(float(r1['delta_u']))} and changed {r1['changed_set_count']} of 100 sets; its BiAlignF1 change was {_fmt(float(r1['delta_bialign_f1']), 5)}. Two-swap increased Utility@8 by {_fmt(float(r2['delta_u']))} and BiAlignF1 by {_fmt(float(r2['delta_bialign_f1']), 5)}. Relative to One-swap, Two-swap added {_fmt(float(r2r1['delta_u']))} Utility@8 with a BiAlignF1 change of {_fmt(float(r2r1['delta_bialign_f1']), 5)}. Direct produced {_fmt(float(direct['delta_u']))} Utility@8 over Raw Dense-8; its increment over Two-swap was {_fmt(float(dr2['delta_u']))}, while the corresponding BiAlignF1 change was {_fmt(float(dr2['delta_bialign_f1']), 5)}.",
        "",
        f"The archived MMR ablation changed {mmr['changed_set_count']} sets. It reduced Utility@8 by {_fmt(float(mmr['delta_u']))} relative to additive Two-swap and changed RCC by {_fmt(float(mmr['delta_rcc']), 5)} and BiAlignF1 by {_fmt(float(mmr['delta_bialign_f1']), 5)}. The pool, OOF scores, $K$, and $r$ are matched, while the configured greedy tie rule differs from the additive global-set tie rule. The MMR row is therefore a bounded set-objective ablation. Its earlier reduction in backend-space pairwise similarity yielded no broader observed-response coverage in this cell.",
        "",
        rf"The joint query analysis contains both utility--correspondence convergence and divergence. Among changed sets, Spearman $\rho(\Delta U,\Delta\mathrm{{BiAlignF1}})$ was {float(r2r1_joint['spearman_delta_u_delta_bialign_changed']):+.3f} for Two-swap minus One-swap and {float(mmr_joint['spearman_delta_u_delta_bialign_changed']):+.3f} for MMR minus additive Two-swap. BiAlignF1 remains an ecological correspondence outcome: naturally occurring replies sample a partial set of observed response directions and provide no normative community-preference label.",
        "",
        r"\paragraph{Answer to RQ2b.}",
        r"A fixed Top-8 budget benefits from expanding the feasible replacement space beyond the historical One-swap policy. In the primary Development100 cell, Two-swap attains the highest observed Utility@8 among the evaluated capacity settings; Direct has a small, inconclusive difference relative to Two-swap. The community-response analysis shows that the gains over Raw Dense-8 preserve and slightly increase bidirectional correspondence, while the incremental Two-swap--One-swap change remains uncertain and RCC intervals include zero. The archived MMR objective lowers textual redundancy but yields no improvement in RCC or BiAlignF1. RQ2b is therefore answered by utility-aware multi-replacement selection with an explicit action-space audit; community correspondence supplies an independent ecological check on the resulting evidence sets.",
        "",
    ])


def run(output_dir: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = _load_config()
    destination = (output_dir or cfg["output_dir"]).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")

    action_dir = cfg["action_space_dir"]
    set_dir = cfg["set_aware_dir"]
    action_paths = {
        "config": action_dir / "config.json",
        "manifest": action_dir / "reproduction_manifest.json",
        "sets": action_dir / "selected_sets.parquet",
        "predictions": action_dir / "oof_candidate_predictions.parquet",
        "qids": action_dir / "query_ids.txt",
    }
    set_paths = {
        "config": set_dir / "config.json",
        "manifest": set_dir / "reproduction_manifest.json",
        "sets": set_dir / "selected_sets.parquet",
    }
    required = [*action_paths.values(), *set_paths.values(), cfg["corpus_source_map"]]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    action_manifest = json.loads(action_paths["manifest"].read_text(encoding="utf-8"))
    set_manifest = json.loads(set_paths["manifest"].read_text(encoding="utf-8"))
    action_cfg = json.loads(action_paths["config"].read_text(encoding="utf-8"))
    set_cfg_record = json.loads(set_paths["config"].read_text(encoding="utf-8"))
    configured_tie_break_match = action_cfg.get("tie_break") == set_cfg_record.get("tie_break")
    if action_manifest.get("status") != "COMPLETE" or set_manifest.get("status") != "COMPLETE":
        raise ValueError("source selection packages are incomplete")
    if action_manifest["invariants"]["frozen_test_read"] or set_manifest["invariants"]["frozen_test_read"]:
        raise ValueError("source package reports frozen-test access")
    if not set_manifest["invariants"]["community_replies_used_in_selection_or_tuning"] is False:
        raise ValueError("community reply separation invariant failed")
    if not set_manifest["invariants"]["same_oof_predictions_as_action_space_repair"]:
        raise ValueError("MMR prediction-source invariant failed")

    qids = [line.strip() for line in action_paths["qids"].read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_n = int(cfg["expected_query_count"])
    if len(qids) != expected_n or len(set(qids)) != expected_n:
        raise ValueError("Development100 query identity changed")
    qid_set = set(qids)

    backend = str(cfg["backend"])
    scorer = str(cfg["scorer"])
    depth = int(cfg["dense_depth"])
    pool_family = str(cfg["pool_family"])
    direct_budget = int(cfg["direct_replacement_budget"])
    unrestricted_budget = int(cfg["unrestricted_check_budget"])
    mmr_budget = int(cfg["mmr_replacement_budget"])
    mmr_lambda = float(cfg["mmr_lambda"])
    tolerance = float(cfg["equality_tolerance"])
    metric_tolerance = float(cfg["metric_recompute_tolerance"])

    # Selection artefacts are fully loaded and checked before hidden replies.
    action_frame = pd.read_parquet(action_paths["sets"])
    action_frame = action_frame[
        (action_frame["selection_kind"] == "learned")
        & (action_frame["backend"] == backend)
        & (action_frame["pool_family"] == pool_family)
        & (action_frame["dense_depth"] == depth)
        & (action_frame["scorer"] == scorer)
        & action_frame["replacement_budget"].isin([0, 1, 2, direct_budget, unrestricted_budget])
    ].copy()
    if len(action_frame) != expected_n * 5:
        raise ValueError(f"primary action set grid incomplete: {len(action_frame)}")

    set_frame = pd.read_parquet(set_paths["sets"])
    set_frame = set_frame[
        (set_frame["backend"] == backend)
        & (set_frame["pool_family"] == pool_family)
        & (set_frame["dense_depth"] == depth)
        & (set_frame["scorer"] == scorer)
        & set_frame["replacement_budget"].isin([1, 2, direct_budget, unrestricted_budget])
        & set_frame["diversity_lambda"].isin([0.0, mmr_lambda])
    ].copy()

    action_lookup = {
        (int(row.replacement_budget), str(row.query_id)): row
        for row in action_frame.itertuples(index=False)
    }
    set_lookup = {
        (int(row.replacement_budget), float(row.diversity_lambda), str(row.query_id)): row
        for row in set_frame.itertuples(index=False)
    }
    for qid in qids:
        for budget in (1, 2, direct_budget, unrestricted_budget):
            action_ids = set(map(str, action_lookup[(budget, qid)].selected_comment_ids))
            additive_ids = set(map(str, set_lookup[(budget, 0.0, qid)].selected_comment_ids))
            if action_ids != additive_ids:
                raise ValueError(f"lambda=0/action mismatch: {qid}/r{budget}")
        if set(map(str, action_lookup[(direct_budget, qid)].selected_comment_ids)) != set(
            map(str, action_lookup[(unrestricted_budget, qid)].selected_comment_ids)
        ):
            raise ValueError(f"Direct r{direct_budget}/r{unrestricted_budget} mismatch: {qid}")

    setaware_cfg = setaware._setaware_resolve_config(setaware.CONFIG_KEY)
    if (
        setaware_cfg["primary_cell"]["backend"] != backend
        or setaware_cfg["primary_cell"]["scorer"] != scorer
        or int(setaware_cfg["primary_cell"]["dense_depth"]) != depth
        or int(setaware_cfg["primary_cell"]["replacement_budget"]) != mmr_budget
        or abs(float(setaware_cfg["primary_diversity_lambda"]) - mmr_lambda) > tolerance
    ):
        raise ValueError("frozen MMR primary cell differs from requested matched cell")

    corpus_rows = json.loads(setaware_cfg["corpus"].read_text(encoding="utf-8"))
    corpus = {str(row["title"]): str(row["text"]) for row in corpus_rows}
    complete_rows, registry = complete_utility_v2_rows(read_jsonl(setaware_cfg["utility_registry"]))
    source_map = _load_source_map(cfg["corpus_source_map"])

    frozen_sets: dict[tuple[str, str], dict[str, Any]] = {}
    for qid in qids:
        raw = action_lookup[(0, qid)]
        methods = {
            "raw_dense8": raw,
            "one_swap": set_lookup[(1, 0.0, qid)],
            "two_swap": set_lookup[(2, 0.0, qid)],
            "direct": set_lookup[(direct_budget, 0.0, qid)],
            "mmr_r2": set_lookup[(mmr_budget, mmr_lambda, qid)],
        }
        raw_ids = set(map(str, raw.selected_comment_ids))
        for method, row in methods.items():
            ids = list(map(str, row.selected_comment_ids))
            if len(ids) != int(cfg["final_k"]) or len(set(ids)) != int(cfg["final_k"]):
                raise ValueError(f"{method}/{qid}: final set is not eight unique comments")
            missing_corpus = sorted(set(ids) - set(corpus))
            missing_labels = sorted(cid for cid in ids if (qid, cid) not in registry)
            missing_source = sorted(set(ids) - set(source_map))
            same_thread = sorted(cid for cid in ids if source_map.get(cid) == qid)
            if missing_corpus or missing_labels or missing_source or same_thread:
                raise ValueError(
                    f"{method}/{qid}: corpus={missing_corpus}, labels={missing_labels}, "
                    f"source={missing_source}, same_thread={same_thread}"
                )
            replacement_count = len(set(ids) - raw_ids)
            recorded = int(row.replacement_count)
            if replacement_count != recorded:
                raise ValueError(f"{method}/{qid}: replacement-count mismatch")
            frozen_sets[(method, qid)] = {
                "selected_comment_ids": ids,
                "selected_utility_at8": _finite(row.selected_utility_at8),
                "replacement_count": replacement_count,
            }

    # Hidden references and their frozen embeddings are opened only here.
    references = community.load_admin_references(cfg["community_reference_dir"])
    if set(references) != qid_set:
        raise ValueError("hidden reply query set differs from frozen selections")
    hidden_reply_count = sum(map(len, references.values()))
    if hidden_reply_count != int(cfg["expected_hidden_reply_count"]):
        raise ValueError("hidden reply count changed")
    bge_embeddings, bge_audit = setaware._load_bge_embeddings(setaware_cfg, corpus, references)

    per_query: list[dict[str, Any]] = []
    for qid in qids:
        raw_ids = set(frozen_sets[("raw_dense8", qid)]["selected_comment_ids"])
        for method in METHOD_ORDER:
            item = frozen_sets[(method, qid)]
            ids = item["selected_comment_ids"]
            if method == "raw_dense8":
                aligned = setaware._alignment(
                    ids, qid, corpus, references, bge_embeddings,
                    float(setaware_cfg["correspondence_threshold"]),
                )
                correspondence_source = "new_posthoc_from_frozen_bge_cache"
            else:
                if method == "one_swap":
                    row = set_lookup[(1, 0.0, qid)]
                elif method == "two_swap":
                    row = set_lookup[(2, 0.0, qid)]
                elif method == "direct":
                    row = set_lookup[(direct_budget, 0.0, qid)]
                else:
                    row = set_lookup[(mmr_budget, mmr_lambda, qid)]
                aligned = {
                    "cra_at8": _finite(row.cra_at8),
                    "rcc_at8": _finite(row.rcc_at8),
                    "bialign_f1_at8": _finite(row.bialign_f1_at8),
                    "best_align_at8": _finite(row.best_align_at8),
                }
                recomputed = setaware._alignment(
                    ids, qid, corpus, references, bge_embeddings,
                    float(setaware_cfg["correspondence_threshold"]),
                )
                for metric in aligned:
                    if abs(aligned[metric] - recomputed[metric]) > metric_tolerance:
                        raise ValueError(f"frozen correspondence mismatch: {method}/{qid}/{metric}")
                correspondence_source = "verified_frozen_set_aware_output"
            per_query.append({
                "query_id": qid,
                "method": method,
                "method_label": _method_label(method).replace("$", ""),
                "selected_ids_json": json.dumps(ids, ensure_ascii=False),
                "utility_at8": item["selected_utility_at8"],
                "cra_at8": aligned["cra_at8"],
                "rcc_at8": aligned["rcc_at8"],
                "bialign_f1_at8": aligned["bialign_f1_at8"],
                "best_align_at8": aligned["best_align_at8"],
                "actual_replacement_count": item["replacement_count"],
                "changed_from_raw": set(ids) != raw_ids,
                "hidden_reply_count": len(references[qid]),
                "correspondence_source": correspondence_source,
            })

    index = _index_rows(pd.DataFrame(per_query))
    rng = np.random.default_rng(int(cfg["bootstrap_seed"]))
    bootstrap_indices = rng.integers(
        0, expected_n, size=(int(cfg["bootstrap_samples"]), expected_n)
    )
    expected_bootstrap_hash = set_cfg_record.get("bootstrap_indices_sha256")
    actual_bootstrap_hash = _object_sha256(bootstrap_indices.tolist())
    if expected_bootstrap_hash and expected_bootstrap_hash != actual_bootstrap_hash:
        raise ValueError("bootstrap identity differs from frozen set-aware analysis")

    paired_rows: list[dict[str, Any]] = []
    for left, right in CONTRASTS:
        changed = [
            set(json.loads(index[(left, qid)]["selected_ids_json"]))
            != set(json.loads(index[(right, qid)]["selected_ids_json"]))
            for qid in qids
        ]
        result: dict[str, Any] = {
            "method": left,
            "method_label": _method_label(left).replace("$", ""),
            "baseline": right,
            "baseline_label": _method_label(right).replace("$", ""),
            "queries": expected_n,
            "changed_set_count": int(sum(changed)),
            "unchanged_set_count": int(expected_n - sum(changed)),
        }
        for metric in METRICS:
            deltas = [
                _finite(index[(left, qid)][metric]) - _finite(index[(right, qid)][metric])
                for qid in qids
            ]
            mean, low, high = _mean_ci(deltas, bootstrap_indices)
            wins, ties, losses = _wtl(deltas, tolerance)
            short = METRIC_SHORT[metric]
            result.update({
                f"delta_{short}": mean,
                f"delta_{short}_ci_low": low,
                f"delta_{short}_ci_high": high,
                f"{short}_wins": wins,
                f"{short}_ties": ties,
                f"{short}_losses": losses,
            })
        paired_rows.append(result)

    quadrant_rows: list[dict[str, Any]] = []
    quadrant_summary: list[dict[str, Any]] = []
    for left, right in QUADRANT_CONTRASTS:
        changed_u: list[float] = []
        changed_c: list[float] = []
        changed_rcc: list[float] = []
        categories: Counter[str] = Counter()
        all_u: list[float] = []
        all_c: list[float] = []
        all_rcc: list[float] = []
        for qid in qids:
            left_row = index[(left, qid)]
            right_row = index[(right, qid)]
            left_ids = set(json.loads(left_row["selected_ids_json"]))
            right_ids = set(json.loads(right_row["selected_ids_json"]))
            du = _finite(left_row["utility_at8"]) - _finite(right_row["utility_at8"])
            dc = _finite(left_row["bialign_f1_at8"]) - _finite(right_row["bialign_f1_at8"])
            drcc = _finite(left_row["rcc_at8"]) - _finite(right_row["rcc_at8"])
            all_u.append(du)
            all_c.append(dc)
            all_rcc.append(drcc)
            if left_ids == right_ids:
                category = "unchanged_set"
            elif du > tolerance and dc > tolerance:
                category = "u_up_c_up"
            elif du > tolerance and dc <= tolerance:
                category = "u_up_c_nonpositive"
            elif du <= tolerance and dc > tolerance:
                category = "u_nonpositive_c_up"
            else:
                category = "u_nonpositive_c_nonpositive"
            categories[category] += 1
            if category != "unchanged_set":
                changed_u.append(du)
                changed_c.append(dc)
                changed_rcc.append(drcc)
            quadrant_rows.append({
                "query_id": qid,
                "method": left,
                "baseline": right,
                "changed_set": left_ids != right_ids,
                "delta_utility_at8": du,
                "delta_bialign_f1_at8": dc,
                "delta_rcc_at8": drcc,
                "category": category,
            })
        changed_n = expected_n - categories["unchanged_set"]
        summary = {
            "method": left,
            "baseline": right,
            "queries": expected_n,
            "changed_set_count": changed_n,
            "unchanged_set_count": categories["unchanged_set"],
            "u_up_c_up_count": categories["u_up_c_up"],
            "u_up_c_nonpositive_count": categories["u_up_c_nonpositive"],
            "u_nonpositive_c_up_count": categories["u_nonpositive_c_up"],
            "u_nonpositive_c_nonpositive_count": categories["u_nonpositive_c_nonpositive"],
            "spearman_delta_u_delta_bialign_all": _rho(all_u, all_c),
            "spearman_delta_u_delta_rcc_all": _rho(all_u, all_rcc),
            "spearman_delta_u_delta_bialign_changed": _rho(changed_u, changed_c),
            "spearman_delta_u_delta_rcc_changed": _rho(changed_u, changed_rcc),
        }
        for key in (
            "u_up_c_up", "u_up_c_nonpositive", "u_nonpositive_c_up",
            "u_nonpositive_c_nonpositive",
        ):
            summary[f"{key}_fraction_changed"] = (
                categories[key] / changed_n if changed_n else None
            )
        quadrant_summary.append(summary)

    method_summary: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        rows = [index[(method, qid)] for qid in qids]
        method_summary.append({
            "method": method,
            "mean_utility_at8": statistics.fmean(_finite(row["utility_at8"]) for row in rows),
            "mean_cra_at8": statistics.fmean(_finite(row["cra_at8"]) for row in rows),
            "mean_rcc_at8": statistics.fmean(_finite(row["rcc_at8"]) for row in rows),
            "mean_bialign_f1_at8": statistics.fmean(_finite(row["bialign_f1_at8"]) for row in rows),
            "mean_best_align_at8": statistics.fmean(_finite(row["best_align_at8"]) for row in rows),
            "changed_from_raw_count": int(sum(bool(row["changed_from_raw"]) for row in rows)),
            "replacement_count_distribution": json.dumps(
                dict(sorted(Counter(int(row["actual_replacement_count"]) for row in rows).items()))
            ),
        })

    with tempfile.TemporaryDirectory(prefix="rq2b_c_", dir=destination.parent) as temp_name:
        out = Path(temp_name)
        _write_csv(out / "rq2b_c_per_query.csv", per_query)
        _write_csv(out / "rq2b_c_paired_contrasts.csv", paired_rows)
        _write_csv(out / "rq2b_c_quadrants.csv", quadrant_rows)
        _write_csv(out / "rq2b_c_quadrant_summary.csv", quadrant_summary)
        _write_csv(out / "rq2b_c_method_summary.csv", method_summary)
        (out / "rq2b_c_table.tex").write_text(
            _build_latex(paired_rows, quadrant_summary), encoding="utf-8"
        )
        (out / "rq2b_c_patch.tex").write_text(
            _build_patch(paired_rows, quadrant_summary), encoding="utf-8"
        )

        pair_lookup = {(row["method"], row["baseline"]): row for row in paired_rows}
        mmr = pair_lookup[("mmr_r2", "two_swap")]
        audit_lines = [
            "# RQ2b set-level community-response correspondence audit", "",
            "## Audit classification", "",
            "- **Already computed:** per-query CRA, RCC, BiAlignF1 and BestAlign for additive r=1/r=2/r=4/r=8 and matched MMR in the frozen set-aware package.",
            "- **Reconstructed without retraining:** Raw Dense-8 correspondence from its frozen IDs and the already frozen BGE-M3 cache; the six requested paired contrasts; joint U/C quadrants.",
            "- **New inference:** none. No model fitting, embedding generation, LLM judging, candidate selection or hyperparameter choice was performed.",
            "- **Missing required artefact:** none for the declared Development100 primary comparison.", "",
            "## Frozen primary cell", "",
            f"- Backend/scorer/pool: `{backend}` / `{scorer}` / Dense M={depth}.",
            f"- K={cfg['final_k']}; additive budgets r=0/1/2/{direct_budget}/{unrestricted_budget}.",
            f"- Matched MMR: M={depth}, r={mmr_budget}, lambda={mmr_lambda:.2f}.",
            f"- Direct identity check: r={direct_budget} and r={unrestricted_budget} agree on {expected_n}/{expected_n} queries.",
            f"- Lambda-zero reproduction: action-space and set-aware additive outputs agree on {expected_n * 4}/{expected_n * 4} primary-cell checks.", "",
            "## Community correspondence implementation", "",
            "- Representation: `BAAI/bge-m3`, frozen revision `5617a9f61b028005a4858fdac845db406aefb181`, CLS pooling, 1024 dimensions, maximum length 512, L2-normalised embeddings.",
            "- Similarity: matrix product of normalised embeddings, equivalent to cosine similarity.",
            "- CRA: mean over selected candidates of each candidate's maximum hidden-reply similarity.",
            "- RCC: mean over hidden replies of each reply's maximum selected-candidate similarity.",
            "- BiAlignF1: harmonic mean of per-query CRA and RCC, with zero for a zero denominator.",
            "- BestAlign: maximum candidate--reply similarity in the query matrix.",
            "- Threshold: tau=.70 applies only to thresholded ReplyCoverage; CRA/RCC/BiAlignF1/BestAlign are unthresholded.",
            "- Aggregation: query means and paired whole-query bootstrap, 5,000 draws, seed 20260810.",
            f"- Equality tolerance: {tolerance:.0e}.", "",
            f"- Frozen-metric recomputation tolerance: {metric_tolerance:.0e}; the largest observed float32 matrix-product difference was below this bound.", "",
            "The code implementation agrees with the Chapter 3 equations. The frozen selection packages are read before the hidden-reference file in this audit. The historical set-aware runner also records `community_used_in_selection=False` and exact lambda-zero reproduction; its source loads the reference package before iterating over MMR sets, so the strongest chronological claim should be attached to this post-hoc audit rather than to the original runner's internal load order.", "",
            "## Integrity checks", "",
            f"- Query set: {expected_n}/{expected_n}; hidden replies: {hidden_reply_count}.",
            "- Every method/query output has eight unique comments, complete utility labels, complete corpus/source mappings and zero same-thread comments.",
            "- All five methods use the same query set. Additive outputs and MMR use the same frozen OOF prediction source.",
            "- MMR and additive r=2 share P12, E5 Small-MLP scores, K=8 and r=2.",
            f"- Configured tie-breaking match: {configured_tie_break_match}. Additive uses `{action_cfg.get('tie_break')}`; MMR uses `{set_cfg_record.get('tie_break')}`. The requested strict single-factor condition therefore fails at the configured tie-rule level, and the MMR row is retained as a qualified archived ablation.", "",
            "## Boundary", "",
            "Hidden replies remain post-hoc ecological references. They do not enter U, scorer fitting, replacement decisions, lambda selection or evidence-set construction. Development100 is the only cohort with the frozen 1,749-reply package.",
        ]
        (out / "rq2b_c_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

        result_lines = [
            "# RQ2b set-level utility and community correspondence", "",
            "## Primary paired results", "",
            "| Contrast | Delta U@8 | Delta RCC | Delta BiAlignF1 | Changed sets |",
            "|---|---:|---:|---:|---:|",
        ]
        for left, right in CONTRASTS:
            row = pair_lookup[(left, right)]
            result_lines.append(
                f"| {_method_label(left).replace('$', '')} - {_method_label(right).replace('$', '')} | "
                f"{_ci_cell(row, 'u')} | {_ci_cell(row, 'rcc', 5)} | "
                f"{_ci_cell(row, 'bialign_f1', 5)} | {row['changed_set_count']}/100 |"
            )
        result_lines.extend([
            "", "## Joint Utility/BiAlignF1 analysis", "",
            "| Contrast | U up / C up | U up / C nonpositive | U nonpositive / C up | U nonpositive / C nonpositive | Unchanged | rho dU/dBiAlign changed | rho dU/dRCC changed |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in quadrant_summary:
            rho_f1 = row["spearman_delta_u_delta_bialign_changed"]
            rho_rcc = row["spearman_delta_u_delta_rcc_changed"]
            result_lines.append(
                f"| {_method_label(str(row['method'])).replace('$', '')} - {_method_label(str(row['baseline'])).replace('$', '')} | "
                f"{row['u_up_c_up_count']} | {row['u_up_c_nonpositive_count']} | "
                f"{row['u_nonpositive_c_up_count']} | {row['u_nonpositive_c_nonpositive_count']} | "
                f"{row['unchanged_set_count']} | "
                f"{('N/A' if rho_f1 is None else f'{float(rho_f1):+.3f}')} | "
                f"{('N/A' if rho_rcc is None else f'{float(rho_rcc):+.3f}')} |"
            )
        result_lines.extend([
            "", "## Interpretation", "",
            "- Replacement capacity increases Utility@8 from Raw through Two-swap. Direct does not improve on Two-swap in this primary cell; the paired difference is small and inconclusive.",
            "- Correspondence changes are small. The paired intervals and query quadrants preserve both convergence and divergence between rubric utility and observed reply directions.",
            f"- The archived MMR arm changes {mmr['changed_set_count']}/100 sets; its RCC change is {_ci_cell(mmr, 'rcc', 5)} and its BiAlignF1 change is {_ci_cell(mmr, 'bialign_f1', 5)}. The redundancy reduction therefore does not convert into broader observed-response coverage. Its configured tie rule differs from additive selection, so this contrast remains a qualified ablation.",
            "- U remains the optimisation and evidence outcome. C remains an independent ecological set-level outcome.", "",
            "## Evidence strength", "",
            "All correspondence conclusions are Development100-only. They are suitable for the Chapter 5 RQ2b mechanism analysis with an explicit nested-subset qualifier. Full CRA/BestAlign intervals, W/T/L and the quadrant audit fit best in the appendix. Test200 and Development300 supply no matching hidden-reply package for this comparison.",
        ])
        (out / "rq2b_c_results.md").write_text("\n".join(result_lines) + "\n", encoding="utf-8")

        source_hashes = {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [*action_paths.values(), *set_paths.values(), cfg["corpus_source_map"]]
        }
        source_hashes[str(setaware_cfg["utility_registry"].relative_to(ROOT))] = sha256(
            setaware_cfg["utility_registry"]
        )
        source_hashes[str(setaware_cfg["bge_cache"].relative_to(ROOT))] = sha256(
            setaware_cfg["bge_cache"]
        )
        source_hashes[str((cfg["community_reference_dir"] / "ADMIN_community_reply_reference_texts.jsonl").relative_to(ROOT))] = sha256(
            cfg["community_reference_dir"] / "ADMIN_community_reply_reference_texts.jsonl"
        )
        config_record = {
            "schema": "rq2b-set-correspondence-config-v1",
            "version": cfg["version"],
            "primary_cell": {
                "backend": backend, "scorer": scorer, "pool_family": pool_family,
                "dense_depth": depth, "final_k": int(cfg["final_k"]),
            },
            "methods": list(METHOD_ORDER),
            "contrasts": [list(item) for item in CONTRASTS],
            "quadrant_contrasts": [list(item) for item in QUADRANT_CONTRASTS],
            "mmr": {"replacement_budget": mmr_budget, "lambda": mmr_lambda},
            "tie_breaks": {
                "additive": action_cfg.get("tie_break"),
                "mmr": set_cfg_record.get("tie_break"),
                "configured_match": configured_tie_break_match,
            },
            "bootstrap": {
                "unit": "query", "samples": int(cfg["bootstrap_samples"]),
                "seed": int(cfg["bootstrap_seed"]), "indices_sha256": actual_bootstrap_hash,
            },
            "equality_tolerance": tolerance,
            "metric_recompute_tolerance": metric_tolerance,
            "bge_correspondence_input": bge_audit,
            "hidden_reply_count": hidden_reply_count,
            "selection_frozen_before_hidden_reference_read_in_this_audit": True,
            "external_calls": 0,
            "new_model_training": False,
            "new_embedding_inference": False,
            "new_llm_judging": False,
        }
        _write_json(out / "config.json", config_record)

        output_names = [
            "rq2b_c_audit.md", "rq2b_c_per_query.csv", "rq2b_c_paired_contrasts.csv",
            "rq2b_c_quadrants.csv", "rq2b_c_quadrant_summary.csv",
            "rq2b_c_method_summary.csv", "rq2b_c_results.md", "rq2b_c_table.tex",
            "rq2b_c_patch.tex", "config.json",
        ]
        manifest = {
            "schema": "rq2b-set-correspondence-reproduction-v1",
            "created_utc": _utc_now(),
            "status": "COMPLETE",
            "implementation": {
                "path": "evaluation/analyze_rq2b_set_correspondence.py",
                "sha256": sha256(Path(__file__)),
            },
            "source_hashes": source_hashes,
            "output_hashes": {name: sha256(out / name) for name in output_names},
            "queries": expected_n,
            "hidden_replies": hidden_reply_count,
            "per_query_rows": len(per_query),
            "paired_contrast_rows": len(paired_rows),
            "quadrant_rows": len(quadrant_rows),
            "bootstrap_indices_sha256": actual_bootstrap_hash,
            "runtime_seconds": time.perf_counter() - started,
            "runtime": {
                "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
                "scipy": scipy.__version__,
            },
            "invariants": {
                "all_sets_size_8_unique": True,
                "all_selected_candidates_have_utility_labels": True,
                "all_selected_candidates_have_source_mapping": True,
                "same_thread_selected_comments": 0,
                "same_query_set_all_methods": True,
                "direct_r4_equals_r8_all_queries": True,
                "lambda_zero_matches_additive_all_primary_checks": True,
                "same_oof_prediction_source": True,
                "mmr_pool_scores_k_and_r_matched": True,
                "mmr_configured_tie_break_matches_additive": configured_tie_break_match,
                "mmr_strict_single_factor_requirement_passed": configured_tie_break_match,
                "hidden_replies_used_for_selection_or_tuning": False,
                "new_training": False,
                "new_embedding_inference": False,
                "new_llm_judging": False,
                "external_calls": 0,
                "frozen_test_read": False,
            },
        }
        _write_json(out / "reproduction_manifest.json", manifest)

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), str(destination))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = run(args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
