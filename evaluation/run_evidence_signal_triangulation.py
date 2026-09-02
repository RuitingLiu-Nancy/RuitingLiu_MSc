#!/usr/bin/env python3
"""Candidate-level triangulation of similarity, utility, and reply correspondence.

The analysis is deliberately post-hoc and development-only.  It joins the
frozen Report 95 M<=50 candidate union to the complete utility-v2 registry,
uses the frozen MiniLM/E5 embedding matrices for query--candidate similarity,
and reuses the hidden-reply BGE-M3 protocol for candidate--reply semantic
correspondence.  Community replies are auxiliary observed-response references,
not preference labels or unique gold answers.

No external API is called.  Frozen test paths are rejected, and the output
directory must not already exist.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

try:
    import configuration as project_config
    from evaluation import community_reply_auxiliary as community
    from evaluation.judgment_completeness import (
        DIMS_V2,
        complete_utility_v2_rows,
    )
    from shared.io_utils import (
        read_csv_rows,
        read_jsonl,
        write_csv_rows,
        write_json,
        write_jsonl,
    )
    from fusion.run_depth_graph_utility_community_frontier import rank_bin
    from evaluation.run_m50_community_frontier_analysis import (
        encode_with_verified_prior_cache,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import configuration as project_config
    from evaluation import community_reply_auxiliary as community
    from evaluation.judgment_completeness import (
        DIMS_V2,
        complete_utility_v2_rows,
    )
    from shared.io_utils import (
        read_csv_rows,
        read_jsonl,
        write_csv_rows,
        write_json,
        write_jsonl,
    )
    from fusion.run_depth_graph_utility_community_frontier import rank_bin
    from evaluation.run_m50_community_frontier_analysis import (
        encode_with_verified_prior_cache,
    )


SIGNAL_PAIRS = (
    ("similarity_utility", "similarity", "utility"),
    ("similarity_community", "similarity", "community"),
    ("utility_community", "utility", "community"),
)
COMMUNITY_INTERPRETATION = (
    "observed community-response correspondence; not community preference, "
    "human quality, or unique gold"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def reject_test_path(path: Path) -> None:
    lowered = "/".join(path.parts).lower()
    forbidden = ("test200", "frozen_test", "confirmatory_test", "test_queries")
    if any(token in lowered for token in forbidden):
        raise ValueError(f"frozen-test-like input forbidden: {path}")


def load_analysis_config(root: Path) -> dict[str, Any]:
    raw = dict(project_config.load()["evidence_signal_triangulation"])
    for key in (
        "output_dir",
        "formal_union",
        "utility_registry",
        "queries",
        "query_admin",
        "corpus",
        "community_reference_dir",
        "community_reference_inventory",
        "community_embedding_prior_dir",
        "dense_memberships",
        "dense_oof_actions",
        "graph_oof_actions",
        "matched_oof_actions",
        "graph_candidate_views",
        "minilm_query_embeddings",
        "minilm_corpus_embeddings",
        "e5_query_embeddings",
        "e5_corpus_embeddings",
    ):
        raw[key] = resolve(root, raw[key])
        reject_test_path(raw[key])
    if bool(raw["allow_external_calls"]):
        raise ValueError("triangulation must remain local-only")
    if bool(raw["allow_frozen_test"]):
        raise ValueError("triangulation must not read frozen test")
    return raw


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _rank_center(values: np.ndarray) -> np.ndarray:
    ranked = rankdata(values, method="average").astype(np.float64)
    return ranked - ranked.mean()


def _center(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    return values - values.mean()


def _query_sufficient(
    rows: list[dict[str, Any]],
    x_field: str,
    y_field: str,
    *,
    ranked: bool,
) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _finite(row[x_field]) and _finite(row[y_field]):
            grouped[str(row["query_id"])].append(row)
    qids: list[str] = []
    sufficient: list[list[float]] = []
    for qid in sorted(grouped):
        items = grouped[qid]
        if len(items) < 3:
            continue
        x = np.asarray([float(row[x_field]) for row in items], dtype=np.float64)
        y = np.asarray([float(row[y_field]) for row in items], dtype=np.float64)
        z = np.asarray([float(row["log_comment_chars"]) for row in items], dtype=np.float64)
        if len(set(x)) < 2 or len(set(y)) < 2:
            continue
        if ranked:
            x, y, z = _rank_center(x), _rank_center(y), _rank_center(z)
        else:
            x, y, z = _center(x), _center(y), _center(z)
        qids.append(qid)
        sufficient.append([
            float(len(items)),
            float(x @ x),
            float(y @ y),
            float(z @ z),
            float(x @ y),
            float(x @ z),
            float(y @ z),
        ])
    return qids, np.asarray(sufficient, dtype=np.float64)


def _correlation_from_sufficient(sufficient: np.ndarray) -> float:
    total = sufficient.sum(axis=0)
    denominator = math.sqrt(total[1] * total[2])
    return float(total[4] / denominator) if denominator > 0 else float("nan")


def _partial_from_sufficient(sufficient: np.ndarray) -> float:
    total = sufficient.sum(axis=0)
    if min(total[1], total[2], total[3]) <= 0:
        return float("nan")
    rxy = total[4] / math.sqrt(total[1] * total[2])
    rxz = total[5] / math.sqrt(total[1] * total[3])
    ryz = total[6] / math.sqrt(total[2] * total[3])
    denominator = math.sqrt(max(0.0, 1.0 - rxz * rxz) * max(0.0, 1.0 - ryz * ryz))
    return float((rxy - rxz * ryz) / denominator) if denominator > 0 else float("nan")


def _cluster_bootstrap_sufficient(
    sufficient: np.ndarray,
    *,
    draws: int,
    seed: int,
    partial: bool,
) -> tuple[float, float]:
    if len(sufficient) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(sufficient),
        np.full(len(sufficient), 1.0 / len(sufficient)),
        size=draws,
    )
    totals = weights @ sufficient
    rxy = totals[:, 4] / np.sqrt(totals[:, 1] * totals[:, 2])
    if partial:
        rxz = totals[:, 5] / np.sqrt(totals[:, 1] * totals[:, 3])
        ryz = totals[:, 6] / np.sqrt(totals[:, 2] * totals[:, 3])
        denominator = np.sqrt((1.0 - rxz * rxz) * (1.0 - ryz * ryz))
        estimates = (rxy - rxz * ryz) / denominator
    else:
        estimates = rxy
    estimates = estimates[np.isfinite(estimates)]
    if not len(estimates):
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def _within_query_rhos(
    rows: list[dict[str, Any]], x_field: str, y_field: str
) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _finite(row[x_field]) and _finite(row[y_field]):
            grouped[str(row["query_id"])].append(row)
    result: dict[str, float] = {}
    for qid, items in grouped.items():
        if len(items) < 3:
            continue
        x = [float(row[x_field]) for row in items]
        y = [float(row[y_field]) for row in items]
        if len(set(x)) < 2 or len(set(y)) < 2:
            continue
        rho = float(spearmanr(x, y).statistic)
        if math.isfinite(rho):
            result[qid] = rho
    return result


def _bootstrap_scalar(
    values: list[float], *, draws: int, seed: int, statistic: str
) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(draws, len(array)))]
    if statistic == "mean":
        estimates = sampled.mean(axis=1)
    elif statistic == "median":
        estimates = np.median(sampled, axis=1)
    else:  # pragma: no cover - internal contract
        raise ValueError(statistic)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def association_statistics(
    rows: list[dict[str, Any]],
    *,
    x_field: str,
    y_field: str,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for ranked, label in ((False, "pearson"), (True, "spearman")):
        qids, sufficient = _query_sufficient(rows, x_field, y_field, ranked=ranked)
        for partial, prefix in ((False, "query_centered"), (True, "length_adjusted_query_centered")):
            estimate = (
                _partial_from_sufficient(sufficient)
                if partial
                else _correlation_from_sufficient(sufficient)
            )
            lo, hi = _cluster_bootstrap_sufficient(
                sufficient,
                draws=draws,
                seed=seed + (101 if ranked else 0) + (211 if partial else 0),
                partial=partial,
            )
            output.append({
                "statistic": f"{prefix}_{label}",
                "estimate": estimate,
                "bootstrap_95ci_lo": lo,
                "bootstrap_95ci_hi": hi,
                "eligible_query_count": len(qids),
                "pair_count": int(sufficient[:, 0].sum()) if len(sufficient) else 0,
                "wins": None,
                "ties": None,
                "losses": None,
            })
    per_query = _within_query_rhos(rows, x_field, y_field)
    values = list(per_query.values())
    for statistic_name, estimator in (
        ("mean_within_query_spearman", statistics.fmean),
        ("median_within_query_spearman", statistics.median),
    ):
        estimate = float(estimator(values)) if values else float("nan")
        lo, hi = _bootstrap_scalar(
            values,
            draws=draws,
            seed=seed + (307 if statistic_name.startswith("mean") else 401),
            statistic="mean" if statistic_name.startswith("mean") else "median",
        )
        output.append({
            "statistic": statistic_name,
            "estimate": estimate,
            "bootstrap_95ci_lo": lo,
            "bootstrap_95ci_hi": hi,
            "eligible_query_count": len(values),
            "pair_count": sum(1 for row in rows if row["query_id"] in per_query),
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        })
    return output


def within_query_summary(
    rows: list[dict[str, Any]],
    *,
    x_field: str,
    y_field: str,
    draws: int,
    seed: int,
) -> dict[str, Any] | None:
    per_query = _within_query_rhos(rows, x_field, y_field)
    values = list(per_query.values())
    if not values:
        return None
    lo, hi = _bootstrap_scalar(values, draws=draws, seed=seed, statistic="mean")
    return {
        "statistic": "mean_within_query_spearman",
        "estimate": statistics.fmean(values),
        "bootstrap_95ci_lo": lo,
        "bootstrap_95ci_hi": hi,
        "eligible_query_count": len(values),
        "pair_count": sum(1 for row in rows if row["query_id"] in per_query),
        "median_within_query_spearman": statistics.median(values),
        "wins": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "losses": sum(value < 0 for value in values),
    }


def _prior_bge_texts(
    *,
    references: dict[str, list[dict]],
    corpus: dict[str, str],
    memberships: list[dict],
    action_rows: list[dict],
) -> list[str]:
    rankings: dict[str, dict[str, list[dict]]] = {
        backend: defaultdict(list) for backend in ("minilm", "e5")
    }
    for row in memberships:
        rankings[str(row["backend"])][str(row["query_id"])].append(row)
    candidate_ids: set[str] = set()
    for backend in rankings:
        for query_id, rows in rankings[backend].items():
            rows.sort(key=lambda item: int(item["rank"]))
            candidate_ids.update(str(item["comment_id"]) for item in rows[:8])
    for row in action_rows:
        if bool(row["acted"]):
            candidate_ids.add(str(row["selected_candidate_id"]))
    texts = [reply["text"] for items in references.values() for reply in items]
    texts.extend(corpus[candidate_id] for candidate_id in sorted(candidate_ids))
    return texts


def _candidate_community_score(
    candidate_vector: np.ndarray,
    replies: list[dict],
    embeddings: dict[str, np.ndarray],
) -> float:
    reply_vectors = np.stack([
        embeddings[community.text_sha(str(reply["text"]))]
        for reply in replies
    ])
    return float((reply_vectors @ candidate_vector).max())


def _length_tertiles(rows: list[dict[str, Any]]) -> tuple[float, float]:
    values = np.asarray([float(row["log_comment_chars"]) for row in rows])
    return tuple(float(value) for value in np.quantile(values, [1 / 3, 2 / 3]))


def _figure_source(
    rows: list[dict[str, Any]], *, draws: int, seed: int
) -> list[dict[str, Any]]:
    specifications = [
        ("S-U", "minilm", "similarity_minilm", "utility"),
        ("S-U", "e5", "similarity_e5", "utility"),
        ("S-C", "minilm", "similarity_minilm", "community_alignment_primary"),
        ("S-C", "e5", "similarity_e5", "community_alignment_primary"),
        ("U-C", "shared", "utility", "community_alignment_primary"),
    ]
    output: list[dict[str, Any]] = []
    for spec_index, (panel, backend, x_field, y_field) in enumerate(specifications):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["query_id"])].append(row)
        binned: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for qid, items in grouped.items():
            x = np.asarray([float(item[x_field]) for item in items])
            y = np.asarray([float(item[y_field]) for item in items])
            x_rank = (rankdata(x, method="average") - 0.5) / len(items)
            y_rank = (rankdata(y, method="average") - 0.5) / len(items)
            for xv, yv in zip(x_rank, y_rank, strict=True):
                bin_index = min(10, int(math.floor(float(xv) * 10)) + 1)
                binned[bin_index][qid].append(float(yv))
        for bin_index in range(1, 11):
            by_query = binned[bin_index]
            query_values = [statistics.fmean(values) for values in by_query.values()]
            lo, hi = _bootstrap_scalar(
                query_values,
                draws=draws,
                seed=seed + spec_index * 100 + bin_index,
                statistic="mean",
            )
            output.append({
                "panel": panel,
                "similarity_backend": backend,
                "x_rank_decile": bin_index,
                "x_rank_midpoint": (bin_index - 0.5) / 10.0,
                "mean_y_within_query_rank": statistics.fmean(query_values),
                "bootstrap_95ci_lo": lo,
                "bootstrap_95ci_hi": hi,
                "eligible_query_count": len(query_values),
                "community_interpretation": COMMUNITY_INTERPRETATION,
            })
    return output


def render_figure(
    source_rows: list[dict[str, Any]], output: Path, *, query_count: int = 100
) -> None:
    panels = ("S-U", "S-C", "U-C")
    labels = {
        "S-U": ("Within-query similarity rank", "Within-query utility rank"),
        "S-C": ("Within-query similarity rank", "Within-query reply-correspondence rank"),
        "U-C": ("Within-query utility rank", "Within-query reply-correspondence rank"),
    }
    colours = {"minilm": "#335C81", "e5": "#D2691E", "shared": "#4F772D"}
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.45), sharey=True)
    for axis, panel in zip(axes, panels, strict=True):
        panel_rows = [row for row in source_rows if row["panel"] == panel]
        for backend in sorted({str(row["similarity_backend"]) for row in panel_rows}):
            rows = sorted(
                [row for row in panel_rows if row["similarity_backend"] == backend],
                key=lambda row: int(row["x_rank_decile"]),
            )
            x = np.asarray([float(row["x_rank_midpoint"]) for row in rows])
            y = np.asarray([float(row["mean_y_within_query_rank"]) for row in rows])
            lo = np.asarray([float(row["bootstrap_95ci_lo"]) for row in rows])
            hi = np.asarray([float(row["bootstrap_95ci_hi"]) for row in rows])
            axis.plot(x, y, marker="o", markersize=3.2, color=colours[backend], label=backend)
            axis.fill_between(x, lo, hi, color=colours[backend], alpha=0.14, linewidth=0)
        axis.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)
        axis.set_title(panel)
        axis.set_xlabel(labels[panel][0])
        axis.grid(axis="y", alpha=0.2)
        if panel != "U-C":
            axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel(labels["S-U"][1])
    axes[1].set_ylabel(labels["S-C"][1])
    axes[2].set_ylabel(labels["U-C"][1])
    fig.suptitle(
        f"Candidate-level evidence-signal triangulation ({query_count}-query development set)",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.005,
        "C denotes observed reply correspondence, not community preference.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Release root used to resolve configured input and output paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the configured output directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.project_root.resolve()
    cfg = load_analysis_config(root)
    if args.output_dir is None:
        args.output_dir = cfg["output_dir"]
    output = args.output_dir.resolve()
    reject_test_path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    _, community_cfg = community.load_config()
    community.reject_test_paths(community_cfg)
    references = community.load_admin_references(cfg["community_reference_dir"])
    valid_reply_ids = {
        qid: {str(reply["reply_id"]) for reply in replies}
        for qid, replies in references.items()
    }
    capped_ids = community.load_capped_reply_ids(community_cfg, valid_reply_ids)
    capped_references = {
        qid: [reply for reply in replies if str(reply["reply_id"]) in set(capped_ids[qid])]
        for qid, replies in references.items()
    }

    formal_rows = read_jsonl(cfg["formal_union"])
    if len(formal_rows) != int(cfg["expected_formal_pairs"]):
        raise ValueError("formal M<=50 union identity changed")
    formal_keys = {
        (str(row["query_id"]), str(row["comment_id"])) for row in formal_rows
    }
    if len(formal_keys) != len(formal_rows):
        raise ValueError("formal union contains duplicate query-candidate pairs")
    if any(bool(row["m100_in_scope"]) for row in formal_rows):
        raise ValueError("M100 entered the formal triangulation union")

    complete_rows, utility_registry = complete_utility_v2_rows(
        read_jsonl(cfg["utility_registry"])
    )
    if len(complete_rows) != int(cfg["expected_complete_registry_rows"]):
        raise ValueError("coverage-complete utility registry identity changed")
    if not formal_keys <= set(utility_registry):
        raise ValueError("formal union lacks complete utility-v2 coverage")

    query_rows = read_csv_rows(cfg["queries"])
    query_ids = sorted(str(row["query_id"]) for row in query_rows)
    if len(query_ids) != int(cfg["expected_queries"]) or len(set(query_ids)) != len(query_ids):
        raise ValueError("dev100 query identity changed")
    if set(query_ids) != set(references):
        raise ValueError("hidden-reply references do not match dev100")
    admin_rows = read_csv_rows(cfg["query_admin"])
    need_group = {
        str(row["query_id"]): str(row["llm_single_multi_label"])
        for row in admin_rows
    }
    if Counter(need_group[qid] for qid in query_ids) != Counter({"single_need": 50, "multi_need": 50}):
        raise ValueError("single/multi dev100 balance changed")

    corpus_rows = json.loads(cfg["corpus"].read_text(encoding="utf-8"))
    corpus_ids = [str(row["title"]) for row in corpus_rows]
    corpus = {str(row["title"]): str(row["text"]) for row in corpus_rows}
    if len(corpus_ids) != int(cfg["expected_corpus_comments"]) or len(corpus) != len(corpus_ids):
        raise ValueError("fixed corpus identity changed")
    if not {key[1] for key in formal_keys} <= set(corpus):
        raise ValueError("formal candidate missing from fixed corpus")

    inventory = {
        str(row["query_id"]): row
        for row in read_jsonl(cfg["community_reference_inventory"])
    }
    if set(inventory) != set(query_ids):
        raise ValueError("community reference inventory identity changed")
    if sum(len(references[qid]) for qid in query_ids) != int(cfg["expected_hidden_replies"]):
        raise ValueError("hidden reply count changed")

    memberships = read_jsonl(cfg["dense_memberships"])
    dense_actions = read_jsonl(cfg["dense_oof_actions"])
    graph_actions = read_jsonl(cfg["graph_oof_actions"])
    matched_actions = read_jsonl(cfg["matched_oof_actions"])
    graph_views = read_jsonl(cfg["graph_candidate_views"])
    graph_by_pair: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"routes": set(), "fixed": False, "residual": False}
    )
    for row in graph_views:
        if not bool(row["native_graph"]) or any(
            bool(row[field]) for field in ("fallback_used", "callback_used", "padding_used")
        ):
            raise ValueError("non-strict Graph candidate entered triangulation provenance")
        key = (str(row["query_id"]), str(row["comment_id"]))
        graph_by_pair[key]["routes"].update(str(value) for value in row["graph_routes"])
        graph_by_pair[key]["fixed"] |= str(row["view_type"]) == "fixed_graph4"
        graph_by_pair[key]["residual"] |= str(row["view_type"]) == "residual_graph4"

    target_texts = [
        reply["text"] for replies in references.values() for reply in replies
    ]
    target_texts.extend(corpus[str(row["comment_id"])] for row in formal_rows)
    prior_texts = _prior_bge_texts(
        references=references,
        corpus=corpus,
        memberships=memberships,
        action_rows=dense_actions + graph_actions + matched_actions,
    )
    embeddings, bge_manifest = encode_with_verified_prior_cache(
        config=community_cfg,
        texts=target_texts,
        prior_texts=prior_texts,
        prior_output=cfg["community_embedding_prior_dir"],
        output=output,
    )

    minilm_query = np.load(cfg["minilm_query_embeddings"])
    minilm_corpus = np.load(cfg["minilm_corpus_embeddings"])
    e5_query = np.load(cfg["e5_query_embeddings"])
    e5_corpus = np.load(cfg["e5_corpus_embeddings"])
    expected_shapes = (
        (minilm_query.shape, (len(query_ids), 384)),
        (minilm_corpus.shape, (len(corpus_ids), 384)),
        (e5_query.shape, (len(query_ids), 768)),
        (e5_corpus.shape, (len(corpus_ids), 768)),
    )
    if any(actual != expected for actual, expected in expected_shapes):
        raise ValueError(f"frozen embedding shape mismatch: {expected_shapes}")
    query_index = {query_id: index for index, query_id in enumerate(query_ids)}
    corpus_index = {comment_id: index for index, comment_id in enumerate(corpus_ids)}

    pair_rows: list[dict[str, Any]] = []
    for formal in sorted(formal_rows, key=lambda row: (str(row["query_id"]), str(row["comment_id"]))):
        query_id = str(formal["query_id"])
        comment_id = str(formal["comment_id"])
        key = (query_id, comment_id)
        judged = utility_registry[key]
        text = corpus[comment_id]
        candidate_vector = embeddings[community.text_sha(text)]
        primary_c = _candidate_community_score(candidate_vector, references[query_id], embeddings)
        capped_c = _candidate_community_score(candidate_vector, capped_references[query_id], embeddings)
        dense_memberships = dict(formal["dense_memberships"])
        minilm_rank = (
            int(dense_memberships["minilm"]["rank"])
            if "minilm" in dense_memberships else None
        )
        e5_rank = (
            int(dense_memberships["e5"]["rank"])
            if "e5" in dense_memberships else None
        )
        graph = graph_by_pair.get(key, {"routes": set(), "fixed": False, "residual": False})
        fixed_graph = bool(formal["fixed_graph4"])
        residual_graph = bool(formal["residual_graph4_memberships"])
        if fixed_graph != bool(graph["fixed"]) or residual_graph != bool(graph["residual"]):
            raise ValueError(f"Graph membership disagreement for {key}")
        graph_membership = (
            "fixed_and_residual" if fixed_graph and residual_graph else
            "fixed_graph4" if fixed_graph else
            "residual_graph4" if residual_graph else
            "no_graph_membership"
        )
        routes = sorted(graph["routes"])
        route_membership = "+".join(routes) if routes else "none"
        pair_rows.append({
            "query_id": query_id,
            "comment_id": comment_id,
            "similarity_minilm": float(
                minilm_query[query_index[query_id]] @ minilm_corpus[corpus_index[comment_id]]
            ),
            "similarity_e5": float(
                e5_query[query_index[query_id]] @ e5_corpus[corpus_index[comment_id]]
            ),
            "utility": float(judged["utility"]),
            **{f"label_{dim}": int(judged[f"label_{dim}"]) for dim in DIMS_V2},
            "community_alignment_primary": primary_c,
            "community_alignment_capped8": capped_c,
            "community_interpretation": COMMUNITY_INTERPRETATION,
            "comment_chars": len(text),
            "comment_words": len(text.split()),
            "log_comment_chars": math.log1p(len(text)),
            "reply_count": len(references[query_id]),
            "reply_count_tier": str(inventory[query_id]["depth_tier"]),
            "need_group": need_group[query_id],
            "minilm_dense_rank": minilm_rank,
            "minilm_dense_rank_bin": rank_bin(minilm_rank) if minilm_rank is not None else "outside_minilm_M50",
            "e5_dense_rank": e5_rank,
            "e5_dense_rank_bin": rank_bin(e5_rank) if e5_rank is not None else "outside_e5_M50",
            "fixed_graph4_membership": fixed_graph,
            "residual_graph4_membership": residual_graph,
            "graph_membership": graph_membership,
            "graph_routes": routes,
            "graph_route_membership": route_membership,
            "m100_in_scope": False,
            "frozen_test_read": False,
        })
    if len(pair_rows) != int(cfg["expected_formal_pairs"]):
        raise AssertionError("pair registry row count changed")

    first_cut, second_cut = _length_tertiles(pair_rows)
    for row in pair_rows:
        value = float(row["log_comment_chars"])
        row["comment_length_tertile"] = (
            "short" if value <= first_cut else "medium" if value <= second_cut else "long"
        )
    write_jsonl(output / "candidate_signal_pair_registry.jsonl", pair_rows)

    draws = int(cfg["bootstrap_draws"])
    seed = int(cfg["bootstrap_seed"])
    overall_rows: list[dict[str, Any]] = []
    community_views = (
        ("all_hidden_replies", "community_alignment_primary"),
        ("capped_frozen_replies_max8", "community_alignment_capped8"),
    )
    association_index = 0
    for community_view, community_field in community_views:
        for pair_name, x_name, y_name in SIGNAL_PAIRS:
            backends = ("minilm", "e5") if "similarity" in (x_name, y_name) else ("shared",)
            for backend in backends:
                x_field = f"similarity_{backend}" if x_name == "similarity" else x_name
                y_field = (
                    f"similarity_{backend}" if y_name == "similarity" else
                    community_field if y_name == "community" else y_name
                )
                for result in association_statistics(
                    pair_rows,
                    x_field=x_field,
                    y_field=y_field,
                    draws=draws,
                    seed=seed + association_index * 1000,
                ):
                    overall_rows.append({
                        "signal_pair": pair_name,
                        "similarity_backend": backend,
                        "community_view": community_view,
                        "community_interpretation": COMMUNITY_INTERPRETATION,
                        **result,
                    })
                association_index += 1
    write_csv_rows(output / "overall_signal_associations.csv", overall_rows)
    write_json(output / "query_bootstrap_ci.json", {
        "schema": "candidate-signal-query-bootstrap-v1",
        "cluster_unit": "query_id",
        "draws": draws,
        "seed": seed,
        "statistics": overall_rows,
    })

    stratum_specs = (
        ("comment_length_tertile", "comment_length_tertile", None),
        ("reply_count_tier", "reply_count_tier", None),
        ("need_group", "need_group", None),
        ("dense_rank_bin", None, "backend_specific"),
        ("fixed_graph4_membership", "fixed_graph4_membership", None),
        ("residual_graph4_membership", "residual_graph4_membership", None),
        ("graph_membership", "graph_membership", None),
        ("graph_route_membership", "graph_route_membership", None),
    )
    stratified_rows: list[dict[str, Any]] = []
    stratum_index = 0
    for pair_name, x_name, y_name in SIGNAL_PAIRS:
        backends = ("minilm", "e5") if "similarity" in (x_name, y_name) else ("shared",)
        for backend in backends:
            x_field = f"similarity_{backend}" if x_name == "similarity" else x_name
            y_field = "community_alignment_primary" if y_name == "community" else y_name
            for stratum_name, generic_field, mode in stratum_specs:
                field = (
                    f"{backend}_dense_rank_bin"
                    if mode == "backend_specific" and backend != "shared"
                    else generic_field
                )
                if field is None:
                    continue
                for level in sorted({str(row[field]) for row in pair_rows}):
                    subset = [row for row in pair_rows if str(row[field]) == level]
                    result = within_query_summary(
                        subset,
                        x_field=x_field,
                        y_field=y_field,
                        draws=draws,
                        seed=seed + 100000 + stratum_index * 100,
                    )
                    stratum_index += 1
                    if result is None:
                        continue
                    stratified_rows.append({
                        "signal_pair": pair_name,
                        "similarity_backend": backend,
                        "community_view": "all_hidden_replies",
                        "stratum": stratum_name,
                        "level": level,
                        "community_interpretation": COMMUNITY_INTERPRETATION,
                        **result,
                    })
    write_csv_rows(output / "stratified_signal_associations.csv", stratified_rows)

    figure_rows = _figure_source(pair_rows, draws=draws, seed=seed + 900000)
    write_csv_rows(output / "evidence_signal_triangle_figure_source.csv", figure_rows)
    render_figure(figure_rows, output / "evidence_signal_triangle_summary.pdf")

    input_paths = [
        cfg[key]
        for key in (
            "formal_union", "utility_registry", "queries", "query_admin", "corpus",
            "community_reference_inventory", "dense_memberships", "dense_oof_actions",
            "graph_oof_actions", "matched_oof_actions", "graph_candidate_views",
            "minilm_query_embeddings", "minilm_corpus_embeddings",
            "e5_query_embeddings", "e5_corpus_embeddings",
        )
    ] + [cfg["community_reference_dir"] / "ADMIN_community_reply_reference_texts.jsonl"]
    output_paths = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema": "evidence-signal-triangulation-dev100-v1",
        "created_utc": utc_now(),
        "status": "CANDIDATE_SIGNAL_TRIANGULATION_COMPLETE",
        "formal_universe": {
            "source": str(cfg["formal_union"]),
            "query_candidate_pairs": len(pair_rows),
            "queries": len(query_ids),
            "candidate_ids": len({row["comment_id"] for row in pair_rows}),
            "candidate_text_hashes": len({community.text_sha(corpus[row["comment_id"]]) for row in pair_rows}),
            "M_values": [8, 12, 20, 50],
            "M100_in_scope": False,
        },
        "signals": {
            "S": "frozen normalized query-corpus cosine, reported separately for MiniLM and E5",
            "U": "canonical six-dimensional LLM-silver utility-v2; unchanged",
            "C_primary": "maximum frozen BGE-M3 cosine to all valid hidden top-level replies",
            "C_sensitivity": "maximum frozen BGE-M3 cosine to capped<=8 frozen reply subset",
            "community_interpretation": COMMUNITY_INTERPRETATION,
        },
        "controls": {
            "query_centering": True,
            "query_cluster_bootstrap": True,
            "comment_length_partial_residual": True,
            "comment_length_tertiles_log1p_chars": [first_cut, second_cut],
            "strata": [value[0] for value in stratum_specs],
        },
        "bge_encoder": bge_manifest,
        "invariants": {
            "frozen_test_read": False,
            "external_api_calls": 0,
            "new_llm_judging": False,
            "utility_v2_modified": False,
            "community_used_for_retrieval_selection_or_tuning": False,
            "existing_outputs_overwritten": False,
        },
        "input_sha256": {str(path.relative_to(root)): sha256(path) for path in input_paths},
        "output_sha256": {path.name: sha256(path) for path in output_paths},
        "runtime": {
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "output_dir": str(output),
        "pairs": len(pair_rows),
        "overall_rows": len(overall_rows),
        "stratified_rows": len(stratified_rows),
        "bge_new_texts": bge_manifest["incrementally_encoded_text_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
