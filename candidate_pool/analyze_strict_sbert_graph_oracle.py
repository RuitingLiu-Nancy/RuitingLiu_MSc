#!/usr/bin/env python3
"""Audit strict SBERT reproduction and estimate graph candidate-pool oracles.

The comparison is intentionally backend-matched:

* D8: SBERT dense ranks 1--8;
* D100: the same SBERT dense ranks 1--100;
* D100+G: D100 union graph candidates produced by the same SBERT index.

Oracle values are *judged-pool lower bounds*.  Missing utility-v2 judgments are
reported as residuals and are never imputed as zero or treated as negatives.
Frozen test paths are rejected.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np

try:
    from evaluation.judgment_completeness import (
        complete_utility_v2_rows,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.judgment_completeness import (
        complete_utility_v2_rows,
    )


def _reject_test(path: Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"frozen test artifact is not allowed: {path}")


def _read_jsonl(path: Path) -> list[dict]:
    _reject_test(path)
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_run(path: Path) -> tuple[dict[str, list[str]], dict[tuple[str, str], str]]:
    runs: dict[str, list[str]] = {}
    texts: dict[tuple[str, str], str] = {}
    for row in _read_jsonl(path):
        qid = str(row["query_id"])
        ids = [str(cid) for cid in row["retrieved_titles"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path}: duplicate ids for query {qid}")
        runs[qid] = ids
        for cid, text in zip(
                ids, row.get("retrieved_texts", []), strict=False):
            texts[(qid, cid)] = str(text)
    return runs, texts


def _legacy_dense_run(path: Path) -> dict[str, list[str]]:
    rows = _read_jsonl(path)
    by_query: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_id"])].append(
            (int(row["rank"]), str(row["comment_id"])))
    return {
        qid: [cid for _, cid in sorted(values)]
        for qid, values in by_query.items()
    }


def _corpus_aliases(path: Path) -> tuple[dict[str, str], dict]:
    _reject_test(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    first_by_text: dict[str, str] = {}
    canonical_by_id: dict[str, str] = {}
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        cid = str(row.get("title") or "")
        text = str(row.get("text") or "")
        canonical = first_by_text.setdefault(text, cid)
        canonical_by_id[cid] = canonical
        duplicate_groups[canonical].append(cid)
    groups = {
        canonical: ids for canonical, ids in duplicate_groups.items()
        if len(ids) > 1
    }
    return canonical_by_id, {
        "corpus_rows": len(rows),
        "unique_texts": len(first_by_text),
        "duplicate_text_rows": len(rows) - len(first_by_text),
        "duplicate_groups": groups,
    }


def dense_reproduction_audit(
    official: dict[str, list[str]],
    legacy: dict[str, list[str]],
    aliases: dict[str, str],
    alias_manifest: dict,
) -> dict:
    if set(official) != set(legacy):
        raise ValueError(
            f"query sets differ: official={len(official)}, legacy={len(legacy)}")
    per_query = []
    for qid in sorted(official):
        left = official[qid][:100]
        right = legacy[qid][:100]
        canonical_right = [aliases.get(cid, cid) for cid in right]
        # Official indexing de-duplicates identical passage text.  Remove
        # repeated canonical aliases while preserving the legacy order.
        canonical_unique = list(dict.fromkeys(canonical_right))[:100]
        first_raw = next(
            (rank + 1 for rank, (a, b) in enumerate(zip(left, right))
             if a != b), None)
        first_canonical = next(
            (rank + 1 for rank, (a, b) in enumerate(
                zip(left, canonical_unique)) if a != b), None)
        per_query.append({
            "query_id": qid,
            "raw_top8_exact": left[:8] == right[:8],
            "raw_top100_exact": left == right,
            "canonical_top8_exact": left[:8] == canonical_unique[:8],
            "canonical_top100_exact": left == canonical_unique,
            "raw_top8_overlap": len(set(left[:8]) & set(right[:8])) / 8,
            "raw_top100_overlap": len(set(left) & set(right)) / 100,
            "canonical_top8_overlap":
                len(set(left[:8]) & set(canonical_unique[:8])) / 8,
            "canonical_top100_overlap":
                len(set(left) & set(canonical_unique)) / 100,
            "first_raw_disagreement_rank": first_raw,
            "first_canonical_disagreement_rank": first_canonical,
        })
    n = len(per_query)
    exact = lambda field: sum(bool(row[field]) for row in per_query)
    mean = lambda field: statistics.fmean(row[field] for row in per_query)
    minimum_canonical_overlap = min(
        row["canonical_top100_overlap"] for row in per_query)
    top8_exact = exact("canonical_top8_exact")
    mean_top100_overlap = mean("canonical_top100_overlap")
    return {
        "schema": "strict-sbert-dense-reproduction-audit-v1",
        "queries": n,
        "raw_top8_exact_queries": exact("raw_top8_exact"),
        "raw_top100_exact_queries": exact("raw_top100_exact"),
        "canonical_top8_exact_queries": exact("canonical_top8_exact"),
        "canonical_top100_exact_queries": exact("canonical_top100_exact"),
        "mean_raw_top8_overlap": mean("raw_top8_overlap"),
        "mean_raw_top100_overlap": mean("raw_top100_overlap"),
        "mean_canonical_top8_overlap": mean("canonical_top8_overlap"),
        "mean_canonical_top100_overlap": mean_top100_overlap,
        "minimum_canonical_top100_overlap": minimum_canonical_overlap,
        "gate_passed": (
            top8_exact == n
            and mean_top100_overlap >= 0.999
            and minimum_canonical_overlap >= 0.99
        ),
        "gate_definition": (
            "all canonicalized Top-8 rankings are position-exact; mean "
            "canonical Top-100 set overlap >= .999; every query overlap >= .99"
        ),
        "gate_interpretation": (
            "operational backend equivalence, not byte/rank-exact Top-100 "
            "reproduction; all graph contrasts use this Official-SBERT D100 "
            "as their matched baseline"
        ),
        "duplicate_aliases": alias_manifest,
        "per_query": per_query,
    }


def _top_judged(
    qid: str, candidates: set[str], qrels: dict[tuple[str, str], dict],
    k: int = 8,
) -> tuple[list[tuple[str, float]], float | None]:
    judged = sorted(
        ((cid, float(qrels[(qid, cid)]["utility"]))
         for cid in candidates if (qid, cid) in qrels),
        key=lambda item: (-item[1], item[0]),
    )
    top = judged[:k]
    return top, (
        statistics.fmean(value for _, value in top) if len(top) == k else None)


def _bootstrap_mean(values: list[float], seed: int = 13) -> list[float] | None:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    samples = rng.choice(array, size=(10_000, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975))]


def oracle_analysis(
    dense: dict[str, list[str]],
    graph_runs: dict[str, dict[str, list[str]]],
    graph_texts: dict[tuple[str, str], str],
    qrels: dict[tuple[str, str], dict],
) -> tuple[dict, list[dict], list[dict]]:
    per_query = []
    residual_rows = []
    for qid in sorted(dense):
        dense8 = set(dense[qid][:8])
        dense100 = set(dense[qid][:100])
        graph_by_route = {
            name: set(run.get(qid, [])[:100])
            for name, run in graph_runs.items()
        }
        graph_union = set().union(*graph_by_route.values())
        graph_unique = graph_union - dense100
        pools = {
            "D8": dense8,
            "D100": dense100,
            "D100_plus_G": dense100 | graph_union,
        }
        oracle = {}
        coverage = {}
        for name, pool in pools.items():
            top, mean = _top_judged(qid, pool, qrels)
            judged = sum((qid, cid) in qrels for cid in pool)
            oracle[name] = {
                "top8": [{"comment_id": cid, "utility": utility}
                         for cid, utility in top],
                "mean_utility_at8": mean,
            }
            coverage[name] = {
                "pool": len(pool), "judged": judged,
                "fraction": judged / len(pool) if pool else None,
            }

        dense8_values = [
            float(qrels[(qid, cid)]["utility"])
            for cid in dense8 if (qid, cid) in qrels
        ]
        graph_unique_judged = [
            (cid, qrels[(qid, cid)])
            for cid in graph_unique if (qid, cid) in qrels
        ]
        graph_unique_useful = [
            (cid, row) for cid, row in graph_unique_judged
            if float(row["utility"]) >= 4.0
            and float(row["label_relevance"]) >= 3
            and float(row["label_usefulness"]) >= 3
            and float(row["label_safety"]) >= 4
        ]
        best_graph = max(
            graph_unique_judged,
            key=lambda item: (float(item[1]["utility"]), item[0]),
            default=None,
        )
        dense_tail = min(dense8_values) if len(dense8_values) == 8 else None
        graph_beats_dense_tail = (
            best_graph is not None and dense_tail is not None
            and float(best_graph[1]["utility"]) > dense_tail
        )
        d100_mean = oracle["D100"]["mean_utility_at8"]
        union_mean = oracle["D100_plus_G"]["mean_utility_at8"]
        graph_beyond_depth_gain = (
            union_mean - d100_mean
            if union_mean is not None and d100_mean is not None else None
        )
        for cid in sorted(graph_unique):
            if (qid, cid) not in qrels:
                routes = sorted(
                    name for name, ids in graph_by_route.items() if cid in ids)
                residual_rows.append({
                    "query_id": qid,
                    "comment_id": cid,
                    "comment_text": graph_texts.get((qid, cid), ""),
                    "routes": routes,
                    "item_type": "strict_sbert_graph_unique_residual",
                })
        per_query.append({
            "query_id": qid,
            "pool_coverage": coverage,
            "oracle": oracle,
            "graph_union_candidates": len(graph_union),
            "graph_unique_beyond_dense100": len(graph_unique),
            "graph_unique_judged": len(graph_unique_judged),
            "graph_unique_residual": len(graph_unique) - len(graph_unique_judged),
            "graph_useful_opportunity": bool(graph_unique_useful),
            "best_graph_unique_comment_id": best_graph[0] if best_graph else None,
            "best_graph_unique_utility":
                float(best_graph[1]["utility"]) if best_graph else None,
            "dense_top8_min_utility": dense_tail,
            "graph_beats_dense_top8_tail": graph_beats_dense_tail,
            "oracle_graph_gain_beyond_dense100": graph_beyond_depth_gain,
        })

    gains = [
        row["oracle_graph_gain_beyond_dense100"] for row in per_query
        if row["oracle_graph_gain_beyond_dense100"] is not None
    ]
    summary = {
        "schema": "strict-sbert-graph-oracle-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "development only; LLM-silver utility-v2",
        "oracle_status": (
            "EXISTENCE_RATES_ARE_LOWER_BOUNDS; D100_UNION_CONTRAST_IS "
            "JUDGED_POOL_DIAGNOSTIC"
        ),
        "unjudged_policy": "reported as residual; never imputed",
        "queries": len(per_query),
        "total_graph_unique_beyond_dense100": sum(
            row["graph_unique_beyond_dense100"] for row in per_query),
        "total_judged_graph_unique": sum(
            row["graph_unique_judged"] for row in per_query),
        "graph_unique_judgment_fraction": (
            sum(row["graph_unique_judged"] for row in per_query)
            / sum(row["graph_unique_beyond_dense100"] for row in per_query)
        ),
        "queries_with_complete_dense8_judgments": sum(
            row["dense_top8_min_utility"] is not None for row in per_query),
        "queries_with_graph_unique_beyond_dense100": sum(
            row["graph_unique_beyond_dense100"] > 0 for row in per_query),
        "queries_with_judged_graph_unique": sum(
            row["graph_unique_judged"] > 0 for row in per_query),
        "queries_with_useful_graph_opportunity": sum(
            row["graph_useful_opportunity"] for row in per_query),
        "queries_graph_beats_dense_top8_tail": sum(
            row["graph_beats_dense_top8_tail"] for row in per_query),
        "queries_with_complete_oracle_contrast": len(gains),
        "mean_oracle_graph_gain_beyond_dense100":
            statistics.fmean(gains) if gains else None,
        "oracle_graph_gain_beyond_dense100_95ci":
            _bootstrap_mean(gains),
        "positive_oracle_graph_gain_queries": sum(gain > 0 for gain in gains),
        "zero_oracle_graph_gain_queries": sum(gain == 0 for gain in gains),
        "negative_oracle_graph_gain_queries": sum(gain < 0 for gain in gains),
        "graph_unique_residual_pairs": len(residual_rows),
        "graph_unique_residual_queries": len({
            row["query_id"] for row in residual_rows}),
        "claim_boundary": (
            "Observed useful/replacement existence cannot be undone by "
            "unjudged candidates and is therefore a lower bound. The "
            "difference between two incomplete judged-pool oracles is not a "
            "lower bound on graph gain. A fully judged equal-budget pool is "
            "required; actual selector effectiveness remains a separate contrast."
        ),
    }
    return summary, per_query, residual_rows


def _round_robin_graph_head(
    route_ids: dict[str, list[str]], dense100: set[str], budget: int,
) -> list[str]:
    """Utility-blind route-balanced graph-only shortlist."""
    filtered = {
        name: [cid for cid in ids if cid not in dense100]
        for name, ids in route_ids.items()
    }
    result: list[str] = []
    offsets = {name: 0 for name in filtered}
    names = sorted(filtered)
    while len(result) < budget:
        progressed = False
        for name in names:
            ids = filtered[name]
            while offsets[name] < len(ids):
                cid = ids[offsets[name]]
                offsets[name] += 1
                if cid in result:
                    continue
                result.append(cid)
                progressed = True
                break
            if len(result) == budget:
                break
        if not progressed:
            break
    return result


def balanced_oracle_pilot(
    dense: dict[str, list[str]],
    dense_texts: dict[tuple[str, str], str],
    query_texts: dict[str, str],
    graph_runs: dict[str, dict[str, list[str]]],
    graph_texts: dict[tuple[str, str], str],
    qrels: dict[tuple[str, str], dict],
    budget: int = 8,
) -> tuple[dict, list[dict], list[dict]]:
    """Freeze D8, matched deep-8 and route-balanced graph-8 without labels."""
    rows = []
    residual = []
    source_counts: dict[str, int] = defaultdict(int)
    graph_sizes = []
    for qid in sorted(dense):
        dense8 = dense[qid][:8]
        deep8 = dense[qid][8:8 + budget]
        route_ids = {
            name: run.get(qid, [])[:100] for name, run in graph_runs.items()
        }
        graph8 = _round_robin_graph_head(
            route_ids, set(dense[qid][:100]), budget)
        graph_sizes.append(len(graph8))
        sources = {
            "dense_top8": dense8,
            "dense_depth_control": deep8,
            "graph_beyond_dense100_route_balanced": graph8,
        }
        for source, ids in sources.items():
            for rank, cid in enumerate(ids, start=1):
                source_counts[source] += 1
                row = {
                    "query_id": qid,
                    "query_text": query_texts[qid],
                    "comment_id": cid,
                    "comment_text": (
                        dense_texts.get((qid, cid))
                        or graph_texts.get((qid, cid), "")
                    ),
                    "candidate_source": source,
                    "source_rank": rank,
                    "already_judged": (qid, cid) in qrels,
                    "selection_used_utility": False,
                }
                rows.append(row)
                if (qid, cid) not in qrels:
                    residual.append({
                        **row,
                        "item_type": "strict_sbert_balanced_oracle_residual",
                    })
    unique_pairs = {
        (row["query_id"], row["comment_id"]) for row in rows}
    residual_pairs = {
        (row["query_id"], row["comment_id"]) for row in residual}
    manifest = {
        "schema": "strict-sbert-balanced-oracle-pilot-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "queries": len(dense),
        "budget_per_source_per_query": budget,
        "selection_rule": {
            "dense_top8": "Official-SBERT ranks 1-8",
            "dense_depth_control":
                f"same Official-SBERT ranks 9-{8 + budget}",
            "graph": (
                "round-robin by route name over no-recognition and "
                "fact-only-no-recognition rankings; exclude all D100 ids; "
                f"deduplicate; stop at {budget}"
            ),
            "utility_used_for_selection": False,
        },
        "source_slots": dict(source_counts),
        "graph_slots_per_query": {
            "min": min(graph_sizes),
            "mean": statistics.fmean(graph_sizes),
            "max": max(graph_sizes),
            "queries_with_full_budget": sum(size == budget for size in graph_sizes),
        },
        "unique_pairs": len(unique_pairs),
        "already_judged_unique_pairs": len(unique_pairs - residual_pairs),
        "residual_unique_pairs": len(residual_pairs),
        "residual_queries": len({qid for qid, _ in residual_pairs}),
        "full_oracle_status": (
            "READY" if not residual_pairs else "BLOCKED_PENDING_RESIDUAL_JUDGMENTS"
        ),
        "planned_contrasts": [
            "Oracle(D8 union dense-depth-control) - Oracle(D8)",
            "Oracle(D8 union graph-budget) - Oracle(D8)",
            "Oracle(D8 union depth union graph) - Oracle(D8 union depth)",
            "P(best graph-budget candidate > weakest D8)",
            "P(best graph-budget candidate > best dense-depth-control candidate)",
        ],
    }
    by_query_source: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list))
    for row in rows:
        by_query_source[row["query_id"]][row["candidate_source"]].append(
            row["comment_id"])
    if residual_pairs:
        manifest["metrics"] = {
            "status": "WITHHELD_UNTIL_ALL_BALANCED_POOL_PAIRS_ARE_JUDGED",
        }
    else:
        per_query_metrics = []
        for qid, sources in sorted(by_query_source.items()):
            base = sources["dense_top8"]
            deep = sources["dense_depth_control"]
            graph = sources["graph_beyond_dense100_route_balanced"]

            def utility(cid):
                return float(qrels[(qid, cid)]["utility"])

            def oracle_mean(ids):
                return statistics.fmean(
                    sorted((utility(cid) for cid in set(ids)), reverse=True)[:8])

            base_mean = statistics.fmean(utility(cid) for cid in base)
            depth_mean = oracle_mean(base + deep)
            graph_mean = oracle_mean(base + graph)
            both_mean = oracle_mean(base + deep + graph)
            depth_threshold = sorted(
                (utility(cid) for cid in set(base + deep)), reverse=True)[7]
            strict_graph_entries = sum(
                utility(cid) > depth_threshold for cid in graph)
            tied_graph_entries = sum(
                utility(cid) == depth_threshold for cid in graph)
            per_query_metrics.append({
                "query_id": qid,
                "base_d8_mean": base_mean,
                "oracle_depth_mean": depth_mean,
                "oracle_graph_mean": graph_mean,
                "oracle_depth_graph_mean": both_mean,
                "depth_gain": depth_mean - base_mean,
                "graph_gain": graph_mean - base_mean,
                "graph_vs_depth": graph_mean - depth_mean,
                "graph_gain_beyond_depth": both_mean - depth_mean,
                "strict_graph_entries_after_depth": strict_graph_entries,
                "tied_graph_entries_at_depth_threshold": tied_graph_entries,
                "best_graph_beats_weakest_d8":
                    max(map(utility, graph)) > min(map(utility, base)),
                "best_graph_beats_best_depth":
                    max(map(utility, graph)) > max(map(utility, deep)),
            })
        metric_fields = (
            "depth_gain", "graph_gain", "graph_vs_depth",
            "graph_gain_beyond_depth",
        )
        strict_entry_counts = [
            row["strict_graph_entries_after_depth"]
            for row in per_query_metrics
        ]
        source_quality = {}
        for source, label in (
            ("dense_top8", "dense_top8"),
            ("dense_depth_control", "dense_depth_control"),
            (
                "graph_beyond_dense100_route_balanced",
                "graph_beyond_dense100_route_balanced",
            ),
        ):
            source_rows = [
                qrels[(qid, cid)]
                for qid, sources in by_query_source.items()
                for cid in sources[source]
            ]
            source_quality[label] = {
                "candidates": len(source_rows),
                "mean_utility": statistics.fmean(
                    float(row["utility"]) for row in source_rows),
                "useful_safe_share": statistics.fmean(
                    float(row["utility"]) >= 4
                    and int(row["label_relevance"]) >= 3
                    and int(row["label_usefulness"]) >= 3
                    and int(row["label_safety"]) >= 4
                    for row in source_rows
                ),
            }
        manifest["metrics"] = {
            "status": "COMPLETE",
            "queries": len(per_query_metrics),
            **{
                field: {
                    "mean": statistics.fmean(
                        row[field] for row in per_query_metrics),
                    "query_bootstrap_95ci": _bootstrap_mean(
                        [row[field] for row in per_query_metrics]),
                    "positive_queries": sum(
                        row[field] > 0 for row in per_query_metrics),
                }
                for field in metric_fields
            },
            "p_best_graph_beats_weakest_d8": statistics.fmean(
                row["best_graph_beats_weakest_d8"]
                for row in per_query_metrics),
            "p_best_graph_beats_best_depth": statistics.fmean(
                row["best_graph_beats_best_depth"]
                for row in per_query_metrics),
            "strict_graph_entries_after_depth": {
                "queries_with_any": sum(count > 0 for count in strict_entry_counts),
                "mean_per_query": statistics.fmean(strict_entry_counts),
                "count_distribution": {
                    str(value): strict_entry_counts.count(value)
                    for value in sorted(set(strict_entry_counts))
                },
            },
            "source_quality": source_quality,
            "per_query": per_query_metrics,
        }
    return manifest, rows, residual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-dense", type=Path, required=True)
    ap.add_argument("--legacy-dense", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--graph-run", type=Path, action="append", default=[])
    ap.add_argument("--utility-registry", type=Path)
    ap.add_argument(
        "--balanced-query-ids", type=Path,
        help="Optional development-only JSONL whose query_id set scopes the balanced pilot.",
    )
    ap.add_argument("--balanced-budget", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    for path in (
        args.official_dense, args.legacy_dense, args.corpus, args.out_dir,
        *args.graph_run,
    ):
        _reject_test(path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dense, dense_texts = _official_run(args.official_dense)
    query_texts = {
        str(row["query_id"]): str(row.get("query_text") or "")
        for row in _read_jsonl(args.official_dense)
    }
    legacy = _legacy_dense_run(args.legacy_dense)
    aliases, alias_manifest = _corpus_aliases(args.corpus)
    audit = dense_reproduction_audit(
        dense, legacy, aliases, alias_manifest)
    audit["inputs"] = {
        "official_dense": {
            "path": str(args.official_dense.resolve()),
            "sha256": _sha256(args.official_dense),
        },
        "legacy_dense": {
            "path": str(args.legacy_dense.resolve()),
            "sha256": _sha256(args.legacy_dense),
        },
        "corpus": {
            "path": str(args.corpus.resolve()), "sha256": _sha256(args.corpus),
        },
    }
    _write_json(args.out_dir / "dense_reproduction_audit.json", audit)
    _write_jsonl(
        args.out_dir / "dense_reproduction_per_query.jsonl",
        audit.pop("per_query"))
    if not args.graph_run:
        print(json.dumps({
            "dense_gate_passed": audit["gate_passed"],
            "canonical_top100_exact_queries":
                audit["canonical_top100_exact_queries"],
            "queries": audit["queries"],
        }, ensure_ascii=False, indent=2))
        return
    if not audit["gate_passed"]:
        raise SystemExit(
            "strict dense reproduction gate failed; graph oracle is blocked")
    if args.utility_registry is None:
        raise ValueError("--utility-registry is required with --graph-run")
    if args.balanced_budget < 1:
        raise ValueError("--balanced-budget must be positive")
    _reject_test(args.utility_registry)
    graph_runs = {}
    graph_texts = {}
    for path in args.graph_run:
        run, texts = _official_run(path)
        graph_runs[path.stem] = run
        graph_texts.update(texts)
    _, qrels = complete_utility_v2_rows(_read_jsonl(args.utility_registry))
    summary, per_query, residual = oracle_analysis(
        dense, graph_runs, graph_texts, qrels)
    summary["inputs"] = {
        "graph_runs": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in args.graph_run
        ],
        "utility_registry": {
            "path": str(args.utility_registry.resolve()),
            "sha256": _sha256(args.utility_registry),
            "complete_pairs": len(qrels),
        },
    }
    _write_json(args.out_dir / "oracle_summary.json", summary)
    _write_jsonl(args.out_dir / "oracle_per_query.jsonl", per_query)
    _write_jsonl(args.out_dir / "graph_unique_residual_pairs.jsonl", residual)
    pilot_dense = dense
    if args.balanced_query_ids is not None:
        _reject_test(args.balanced_query_ids)
        pilot_ids = {
            str(row["query_id"]) for row in _read_jsonl(args.balanced_query_ids)
        }
        missing = pilot_ids - set(dense)
        if missing:
            raise ValueError(
                f"balanced query file contains {len(missing)} unknown ids")
        pilot_dense = {qid: dense[qid] for qid in sorted(pilot_ids)}
    pilot_manifest, pilot_rows, pilot_residual = balanced_oracle_pilot(
        pilot_dense, dense_texts, query_texts, graph_runs, graph_texts, qrels,
        budget=args.balanced_budget)
    pilot_manifest["query_scope"] = (
        str(args.balanced_query_ids.resolve())
        if args.balanced_query_ids is not None else "all strict development queries"
    )
    pilot_manifest["inputs"] = summary["inputs"]
    _write_json(
        args.out_dir / "balanced_oracle_pilot_manifest.json", pilot_manifest)
    _write_jsonl(
        args.out_dir / "balanced_oracle_candidate_pool.jsonl", pilot_rows)
    _write_jsonl(
        args.out_dir / "balanced_oracle_residual_pairs.jsonl", pilot_residual)
    summary["balanced_oracle_pilot"] = pilot_manifest
    _write_json(args.out_dir / "oracle_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
