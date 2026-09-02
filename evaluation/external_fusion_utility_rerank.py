"""Utility reranking on the frozen partial-official route-balanced pool.

This adapter is intentionally separate from the internal evidence-pool runner:
the official HippoRAG ranks come from precomputed external JSONL routes rather
than the canonical project arms.  Training and metrics are nevertheless reused
from the canonical grouped LambdaMART and IR implementations.

The default completeness gate refuses partial annotation snapshots.  Frozen
test inputs are rejected.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import configuration as config
from evaluation.ir_metrics import graded_ndcg_at, ndcg_at, recall_at
from evaluation.statistics import bootstrap_ci
from evaluation.quality_diversity_rerank import (
    _fit_lambdamart_fold, _oof_lambdamart,
)
from data_preparation.sampling.sample_human_annotation_candidates import _load_cards


M1 = ("dense_rr", "dense_present", "fusion_rr", "fusion_present")
M2 = M1 + ("official_graph_rr", "official_graph_present",
           "graph_exclusive", "promoted_by_fusion")
M3_EXTRA = ("kg2_h1_rr", "kg2_h1_present", "qafd_h1_rr", "qafd_h1_present",
            "qafd_h2_rr", "qafd_h2_present", "graph_route_count",
            "qafd_only_vs_frozen")
M3 = M2 + M3_EXTRA


def _read_admin(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _assemble(admin: list[dict], route_df: pd.DataFrame, labelled: list[dict]):
    route = {(str(r.query_id), str(r.comment_id)): r._asdict()
             for r in route_df.itertuples(index=False)}
    ratings = {(str(r["query_id"]), str(r["comment_id"])): r for r in labelled}
    rows, missing = [], []
    for source in admin:
        key = (str(source["canonical_query_id"]), str(source["comment_id"]))
        if key not in ratings:
            missing.append(key); continue
        if key not in route:
            raise ValueError(f"missing route feature key: {key}")
        dense_rank = int(source.get("dense_rank") or 0)
        fusion_rank = int(source.get("fusion_rank") or 0)
        row = {
            "query_id": key[0], "comment_id": key[1],
            "utility": float(ratings[key]["utility"]),
            "community_gold": int(ratings[key].get("community_gold", 0)),
            "reasonable": float(float(ratings[key]["label_relevance"]) >= 3 and
                                float(ratings[key]["label_usefulness"]) >= 3 and
                                float(ratings[key]["label_safety"]) >= 4),
            "dense_rr": 1.0 / dense_rank if dense_rank else 0.0,
            "dense_present": float(bool(dense_rank)),
            "fusion_rr": 1.0 / fusion_rank if fusion_rank else 0.0,
            "fusion_present": float(bool(fusion_rank)),
            "dense_rank_raw": dense_rank, "fusion_rank_raw": fusion_rank,
            "official_graph_rr": float(route[key]["official_graph_rr"]),
            "official_graph_present": float(route[key]["official_graph_present"]),
            "graph_exclusive": float(int(source.get("graph_exclusive_at_k") or 0)),
            "promoted_by_fusion": float(int(source.get("promoted_by_fusion_at_k") or 0)),
        }
        row.update({name: float(route[key][name]) for name in M3_EXTRA})
        rows.append(row)
    return rows, missing


def _evaluate_utility_rankings(
    rows: list[dict], scores: list[float], k: int,
) -> dict[str, dict[str, float]]:
    by_q = defaultdict(list)
    for row, score in zip(rows, scores):
        by_q[row["query_id"]].append((row, float(score)))
    result = {}
    for qid, values in by_q.items():
        ranked = [r for r, _ in sorted(values, key=lambda x: (-x[1], x[0]["comment_id"]))]
        gains = {r["comment_id"]: r["utility"] for r, _ in values}
        picked = ranked[:k]
        result[qid] = {
            "utility_ndcg_at_k": graded_ndcg_at(
                [r["comment_id"] for r in ranked], gains, k),
            "utility_mean": float(np.mean([r["utility"] for r in picked])),
            "reasonable_share": float(np.mean([r["reasonable"] for r in picked])),
        }
    return result


def _score_rankings(rows: list[dict], rankings: dict[str, list[str]], k: int) -> dict:
    """Evaluate already materialised rankings against continuous utility."""
    by_q = defaultdict(list)
    for row in rows:
        by_q[row["query_id"]].append(row)
    result = {}
    for qid, values in by_q.items():
        gains = {r["comment_id"]: r["utility"] for r in values}
        ranked = rankings[qid]
        picked = [next(r for r in values if r["comment_id"] == cid) for cid in ranked[:k]]
        result[qid] = {
            "utility_ndcg_at_k": graded_ndcg_at(ranked, gains, k),
            "utility_mean": float(np.mean([r["utility"] for r in picked])),
            "reasonable_share": float(np.mean([r["reasonable"] for r in picked])),
        }
    return result


def _score_rankings_from_scores(rows: list[dict], scores: list[float]) -> dict[str, list[str]]:
    by_q = defaultdict(list)
    for row, score in zip(rows, scores):
        by_q[row["query_id"]].append((row, float(score)))
    return {
        qid: [row["comment_id"] for row, _ in sorted(
            values, key=lambda item: (-item[1], item[0]["comment_id"]))]
        for qid, values in by_q.items()
    }


def _raw_rankings(rows: list[dict], route: str) -> dict[str, list[str]]:
    """Materialise route order within the frozen, fully judged candidate pool."""
    by_q = defaultdict(list)
    for row in rows:
        by_q[row["query_id"]].append(row)
    if route == "dense":
        key = lambda row: (int(row["dense_rank_raw"]) or 10**9, row["comment_id"])
    elif route == "frozen_fusion":
        key = lambda row: (int(row["fusion_rank_raw"]) or 10**9,
                           int(row["dense_rank_raw"]) or 10**9, row["comment_id"])
    elif route == "official_graph":
        key = lambda row: (-float(row["official_graph_rr"]), row["comment_id"])
    else:
        raise ValueError(f"unknown raw route: {route}")
    return {qid: [row["comment_id"] for row in sorted(values, key=key)]
            for qid, values in by_q.items()}


def _residual_rankings(
    rows: list[dict], scores: list[float], k: int, budget: int,
) -> dict[str, list[str]]:
    by_q = defaultdict(list)
    for row, score in zip(rows, scores):
        by_q[row["query_id"]].append((row, float(score)))
    result = {}
    for qid, values in by_q.items():
        baseline = [row for row, _ in sorted(values, key=lambda item: (
            int(item[0]["fusion_rank_raw"]) or 10**9,
            int(item[0]["dense_rank_raw"]) or 10**9, item[0]["comment_id"]))]
        protected = baseline[:max(0, k - budget)]
        protected_ids = {row["comment_id"] for row in protected}
        residual = [row for row, _ in sorted(
            values, key=lambda item: (-item[1], item[0]["comment_id"]))
                    if row["comment_id"] not in protected_ids]
        ranked = protected + residual[:budget]
        ranked_ids = {row["comment_id"] for row in ranked}
        ranked += [row for row in baseline if row["comment_id"] not in ranked_ids]
        result[qid] = [row["comment_id"] for row in ranked]
    return result


def _paired(left: dict, right: dict, metric: str, seed: int) -> dict:
    ids = sorted(set(left) & set(right))
    delta = [left[q][metric] - right[q][metric] for q in ids]
    lo, hi = bootstrap_ci(delta, n_boot=5000, seed=seed)
    return {"queries": len(ids), "mean_delta": float(np.mean(delta)),
            "bootstrap_95_ci": [lo, hi],
            "positive_query_share": float(np.mean(np.asarray(delta) > 0))}


def _residual_evaluate(rows: list[dict], scores: list[float], k: int, budget: int):
    """Keep the external fused backbone and expose only ``budget`` residual slots."""
    rankings = _residual_rankings(rows, scores, k, budget)
    return _score_rankings(rows, rankings, k)


def _nested_residual(rows: list[dict], settings: dict, k: int, return_rankings: bool = False):
    """Outer-heldout evaluation; residual budget selected by inner OOF only."""
    from sklearn.model_selection import GroupKFold
    X = np.asarray([[float(r[n]) for n in M3] for r in rows])
    y = np.asarray([max(0, min(6, int(round(r["utility"])) - 1)) for r in rows])
    groups = np.asarray([r["query_id"] for r in rows])
    group_ids = {q: i for i, q in enumerate(sorted(set(groups)))}
    group_array = np.asarray([group_ids[q] for q in groups], dtype=np.uint32)
    outer = GroupKFold(n_splits=min(settings["folds"], len(group_ids)))
    final, final_rankings, choices = {}, {}, []
    for fold, (train, held) in enumerate(outer.split(X, y, group_array)):
        train_rows = [rows[i] for i in train]
        inner_scores = _oof_lambdamart(
            train_rows, M3, [r["utility"] for r in train_rows],
            [r["query_id"] for r in train_rows], **settings)
        candidates = {}
        base_reasonable = None
        for budget in (0, 1, 2):
            met = _residual_evaluate(train_rows, inner_scores, k, budget)
            ndcg = float(np.mean([v["utility_ndcg_at_k"] for v in met.values()]))
            reasonable = float(np.mean([v["reasonable_share"] for v in met.values()]))
            if budget == 0: base_reasonable = reasonable
            candidates[budget] = (ndcg, reasonable)
        eligible = {b: v for b, v in candidates.items()
                    if v[1] >= float(base_reasonable) - 0.005}
        chosen = max(eligible, key=lambda b: (eligible[b][0], -b))
        held_scores = _fit_lambdamart_fold(
            X, y, group_array, train, held, settings["estimators"],
            settings["learning_rate"], settings["max_depth"],
            settings["pairs_per_sample"], settings["seed"] + fold)
        held_rows = [rows[i] for i in held]
        held_rankings = _residual_rankings(held_rows, held_scores.tolist(), k, chosen)
        final_rankings.update(held_rankings)
        final.update(_score_rankings(held_rows, held_rankings, k))
        choices.append({"outer_fold": fold, "selected_budget": chosen,
                        "inner_candidates": candidates})
    if return_rankings:
        return final, choices, final_rankings
    return final, choices


def _gold_rule_sets(
    rows: list[dict], thresholds: tuple[float, ...], top_ns: tuple[int, ...],
) -> dict[str, dict[str, set[str]]]:
    """Construct separate binary gold sets without merging their estimands."""
    by_q = defaultdict(list)
    for row in rows:
        by_q[row["query_id"]].append(row)
    rules: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for qid, values in by_q.items():
        community = {row["comment_id"] for row in values if row["community_gold"]}
        ranked_utility = sorted(values, key=lambda row: (-row["utility"], row["comment_id"]))
        rules["community"][qid] = community
        for threshold in thresholds:
            high = {row["comment_id"] for row in values if row["utility"] >= threshold}
            suffix = f"{threshold:g}"
            rules[f"utility_ge_{suffix}"][qid] = high
            rules[f"community_intersection_ge_{suffix}"][qid] = community & high
        for top_n in top_ns:
            rules[f"utility_top_{top_n}"][qid] = {
                row["comment_id"] for row in ranked_utility[:top_n]}
    return dict(rules)


def evaluate_gold_sensitivity(
    rows: list[dict], system_rankings: dict[str, dict[str, list[str]]], k: int,
    thresholds: tuple[float, ...], top_ns: tuple[int, ...], seed: int,
    gold_rules: dict[str, dict[str, set[str]]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Score fixed rankings under alternative operational definitions of gold."""
    gold_rules = gold_rules or _gold_rule_sets(rows, thresholds, top_ns)
    score_rows, per_query = [], {}
    for rule, gold_by_q in gold_rules.items():
        for system, rankings in system_rankings.items():
            values = []
            for qid, gold in gold_by_q.items():
                if not gold or qid not in rankings:
                    continue
                ranked = rankings[qid]
                values.append({"query_id": qid,
                               "ndcg": float(ndcg_at(ranked, gold, k)),
                               "recall": float(recall_at(ranked, gold, k))})
            per_query[(rule, system)] = values
            score_rows.append({
                "gold_rule": rule, "system": system, "queries": len(values),
                f"nDCG@{k}": (float(np.mean([row["ndcg"] for row in values]))
                                if values else float("nan")),
                f"Recall@{k}": (float(np.mean([row["recall"] for row in values]))
                                  if values else float("nan")),
            })
    scores = pd.DataFrame(score_rows)
    comparisons = []
    wins = {system: 0 for system in system_rankings}
    ranks = {system: [] for system in system_rankings}
    for rule, group in scores.groupby("gold_rule", sort=False):
        group = group[group["queries"] > 0]
        if group.empty:
            continue
        ordered = group.sort_values([f"nDCG@{k}", "system"], ascending=[False, True])
        best = ordered.iloc[0]; wins[str(best["system"])] += 1
        for rank, system in enumerate(ordered["system"], start=1):
            ranks[str(system)].append(rank)
        if len(ordered) < 2:
            continue
        runner = ordered.iloc[1]
        left = {row["query_id"]: row["ndcg"]
                for row in per_query[(rule, str(best["system"]))]}
        right = {row["query_id"]: row["ndcg"]
                 for row in per_query[(rule, str(runner["system"]))]}
        shared = sorted(set(left) & set(right))
        delta = [left[qid] - right[qid] for qid in shared]
        lo, hi = bootstrap_ci(delta, n_boot=5000, seed=seed)
        comparisons.append({
            "gold_rule": rule, "best": str(best["system"]),
            "runner_up": str(runner["system"]), "queries": len(shared),
            f"delta_nDCG@{k}": float(np.mean(delta)),
            "bootstrap_95_ci_low": lo, "bootstrap_95_ci_high": hi,
        })
    robustness = {
        system: {"rule_wins": wins[system],
                 "mean_rank": float(np.mean(ranks[system])) if ranks[system] else None,
                 "rules": len(ranks[system])}
        for system in system_rankings
    }
    reference_rows = []
    for reference in ("raw_dense", "frozen_fusion"):
        if reference not in system_rankings:
            continue
        for rule in gold_rules:
            reference_values = {row["query_id"]: row["ndcg"]
                                for row in per_query[(rule, reference)]}
            for system in system_rankings:
                if system == reference:
                    continue
                system_values = {row["query_id"]: row["ndcg"]
                                 for row in per_query[(rule, system)]}
                shared = sorted(set(system_values) & set(reference_values))
                if not shared:
                    continue
                delta = [system_values[qid] - reference_values[qid] for qid in shared]
                lo, hi = bootstrap_ci(delta, n_boot=5000, seed=seed)
                reference_rows.append({
                    "gold_rule": rule, "system": system, "reference": reference,
                    "queries": len(shared), f"delta_nDCG@{k}": float(np.mean(delta)),
                    "bootstrap_95_ci_low": lo, "bootstrap_95_ci_high": hi,
                })
    return scores, pd.DataFrame(comparisons), pd.DataFrame(reference_rows), robustness


def analyse_rule_disagreement(
    rows: list[dict], thresholds: tuple[float, ...], top_ns: tuple[int, ...],
    entropy_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, set[str]]]]:
    """Describe disagreement among primitive gold rules.

    The rules are operational definitions, not independent annotators.  We
    therefore report raw subset disagreement and entropy without kappa/alpha.
    Intersection rules are excluded from the vote set because they duplicate
    their community and threshold parents.
    """
    gold_rules = _gold_rule_sets(rows, thresholds, top_ns)
    primitive = ["community"]
    primitive += [f"utility_ge_{value:g}" for value in thresholds]
    primitive += [f"utility_top_{value}" for value in top_ns]
    membership_rows = []
    for row in rows:
        qid, comment_id = row["query_id"], row["comment_id"]
        votes = {
            rule: int(comment_id in gold_rules[rule].get(qid, set()))
            for rule in primitive
        }
        positive_count = sum(votes.values())
        share = positive_count / len(primitive)
        entropy = 0.0 if share in {0.0, 1.0} else (
            -share * math.log2(share) - (1.0 - share) * math.log2(1.0 - share))
        membership_rows.append({
            "query_id": qid, "comment_id": comment_id,
            "utility": row["utility"], "positive_rule_count": positive_count,
            "positive_rule_share": share, "binary_entropy": entropy,
            "high_disagreement": int(entropy >= entropy_threshold), **votes,
        })
    membership = pd.DataFrame(membership_rows)
    subset_rows = []
    for subset_size in range(2, len(primitive) + 1):
        for subset in combinations(primitive, subset_size):
            counts = membership[list(subset)].sum(axis=1)
            mixed = (counts > 0) & (counts < subset_size)
            row = {
                "subset_size": subset_size, "rules": "|".join(subset),
                "candidates": len(membership), "disagreement_count": int(mixed.sum()),
                "disagreement_share": float(mixed.mean()),
                "unanimous_high_share": float((counts == subset_size).mean()),
                "unanimous_low_share": float((counts == 0).mean()),
            }
            if subset_size == 2:
                left, right = subset
                union = ((membership[left] == 1) | (membership[right] == 1)).sum()
                overlap = ((membership[left] == 1) & (membership[right] == 1)).sum()
                row["positive_jaccard"] = float(overlap / union) if union else 0.0
            subset_rows.append(row)
    return membership, pd.DataFrame(subset_rows), gold_rules


def analyse_model_disagreement(
    rows: list[dict], system_rankings: dict[str, dict[str, list[str]]], k: int,
    rank_range_threshold: int, topk_entropy_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe candidate-level rank and top-k disagreement across systems."""
    systems = list(system_rankings)
    row_keys = {(row["query_id"], row["comment_id"]): row for row in rows}
    rank_maps = {
        system: {qid: {comment_id: rank for rank, comment_id in enumerate(ranked, 1)}
                 for qid, ranked in by_q.items()}
        for system, by_q in system_rankings.items()
    }
    membership_rows = []
    for (qid, comment_id), source in row_keys.items():
        ranks = {system: rank_maps[system][qid][comment_id] for system in systems}
        top_votes = {system: int(rank <= k) for system, rank in ranks.items()}
        share = sum(top_votes.values()) / len(systems)
        entropy = 0.0 if share in {0.0, 1.0} else (
            -share * math.log2(share) - (1.0 - share) * math.log2(1.0 - share))
        rank_values = list(ranks.values())
        membership_rows.append({
            "query_id": qid, "comment_id": comment_id, "utility": source["utility"],
            "community_gold": source["community_gold"],
            "rank_min": min(rank_values), "rank_max": max(rank_values),
            "rank_range": max(rank_values) - min(rank_values),
            "rank_std": float(np.std(rank_values, ddof=0)),
            "topk_vote_count": sum(top_votes.values()), "topk_vote_share": share,
            "topk_binary_entropy": entropy,
            "high_disagreement": int(
                max(rank_values) - min(rank_values) >= rank_range_threshold and
                entropy >= topk_entropy_threshold),
            **{f"rank__{system}": rank for system, rank in ranks.items()},
            **{f"top{k}__{system}": vote for system, vote in top_votes.items()},
        })
    membership = pd.DataFrame(membership_rows)
    subset_rows = []
    for subset_size in range(2, len(systems) + 1):
        for subset in combinations(systems, subset_size):
            columns = [f"top{k}__{system}" for system in subset]
            counts = membership[columns].sum(axis=1)
            mixed = (counts > 0) & (counts < subset_size)
            row = {
                "subset_size": subset_size, "systems": "|".join(subset),
                "candidates": len(membership), "topk_disagreement_count": int(mixed.sum()),
                "topk_disagreement_share": float(mixed.mean()),
                "unanimous_in_topk_share": float((counts == subset_size).mean()),
                "unanimous_out_topk_share": float((counts == 0).mean()),
            }
            if subset_size == 2:
                left = membership[f"rank__{subset[0]}"]
                right = membership[f"rank__{subset[1]}"]
                row["mean_absolute_rank_gap"] = float((left - right).abs().mean())
            subset_rows.append(row)
    return membership, pd.DataFrame(subset_rows)


def consensus_core_sensitivity(
    rows: list[dict], system_rankings: dict[str, dict[str, list[str]]], k: int,
    thresholds: tuple[float, ...], top_ns: tuple[int, ...], seed: int,
    membership: pd.DataFrame, full_gold_rules: dict[str, dict[str, set[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Re-score after removing predeclared high-disagreement candidates.

    Gold memberships and top-N membership are frozen *before* removal.  This
    avoids silently promoting the next candidate into a top-N gold set.
    """
    excluded = {
        (str(row.query_id), str(row.comment_id))
        for row in membership[membership["high_disagreement"] == 1].itertuples()
    }
    filtered_rows = [row for row in rows
                     if (row["query_id"], row["comment_id"]) not in excluded]
    filtered_rankings = {
        system: {
            qid: [comment_id for comment_id in ranked
                  if (qid, comment_id) not in excluded]
            for qid, ranked in rankings.items()
        }
        for system, rankings in system_rankings.items()
    }
    filtered_gold = {
        rule: {
            qid: {comment_id for comment_id in gold
                  if (qid, comment_id) not in excluded}
            for qid, gold in by_q.items()
        }
        for rule, by_q in full_gold_rules.items()
    }
    scores, comparisons, _, robustness = evaluate_gold_sensitivity(
        filtered_rows, filtered_rankings, k, thresholds, top_ns, seed,
        gold_rules=filtered_gold)
    metric = f"nDCG@{k}"
    macro = scores[scores["queries"] > 0].groupby("system")[metric].mean()
    summary = {
        "excluded_candidates": len(excluded), "retained_candidates": len(filtered_rows),
        "excluded_share": len(excluded) / len(rows) if rows else 0.0,
        "mean_retained_candidates_per_query": (
            len(filtered_rows) / len({row["query_id"] for row in rows}) if rows else 0.0),
        "macro_mean_ndcg_across_rules": {str(system): float(value)
                                           for system, value in macro.items()},
        "robustness": robustness,
        "claim_guard": (
            "Outcome-informed exclusion is a consensus-core sensitivity only; "
            "it must not replace the full-pool primary estimate."),
    }
    return (scores, membership[membership["high_disagreement"] == 1].copy(),
            comparisons, summary)


def main() -> None:
    cfg = config.load().get("literature_module_ablation", {})
    ap = argparse.ArgumentParser()
    ap.add_argument("--admin-pool", type=Path, required=True)
    ap.add_argument("--route-features", type=Path, required=True)
    ap.add_argument("--study-db", type=Path, required=True)
    ap.add_argument("--heldout", type=Path,
                    help="validation CSV; required to include community/intersection rules")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="diagnostic only; incomplete-query results are not reportable")
    args = ap.parse_args()
    if any("test" in p.name.lower() for p in
           (args.admin_pool, args.route_features, args.study_db) +
           ((args.heldout,) if args.heldout else ())):
        raise ValueError("frozen test inputs are not accepted")
    admin = _read_admin(args.admin_pool)
    route_df = pd.read_parquet(args.route_features)
    rows, missing = _assemble(admin, route_df, _load_cards(args.study_db, args.heldout))
    if args.heldout and not any(row["community_gold"] for row in rows):
        raise ValueError(
            "--heldout supplied but zero pooled community gold matched; "
            "use the validation heldout CSV containing gold_comment_ids")
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"utility completeness gate failed: {len(rows)}/{len(admin)} rated; "
            f"{len(missing)} missing")
    groups = [r["query_id"] for r in rows]
    target = [r["utility"] for r in rows]
    settings = {
        "folds": int(cfg.get("structural_oof_folds", 5)),
        "estimators": int(cfg.get("structural_oof_estimators", 200)),
        "learning_rate": float(cfg.get("structural_oof_learning_rate", 0.05)),
        "max_depth": int(cfg.get("structural_oof_max_depth", 3)),
        "pairs_per_sample": int(cfg.get("structural_oof_pairs_per_sample", 8)),
        "seed": int(cfg.get("seed", 20260712)),
    }
    feature_sets = {"M1_dense_fusion": M1, "M2_official_graph": M2,
                    "M3_literature_routes": M3}
    metrics, system_rankings = {}, {
        "raw_dense": _raw_rankings(rows, "dense"),
        "frozen_fusion": _raw_rankings(rows, "frozen_fusion"),
        "raw_official_graph": _raw_rankings(rows, "official_graph"),
    }
    metrics["raw_dense"] = _score_rankings(rows, system_rankings["raw_dense"], args.k)
    metrics["raw_official_graph"] = _score_rankings(
        rows, system_rankings["raw_official_graph"], args.k)
    metrics["frozen_route_baseline"] = _residual_evaluate(
        rows, [0.0] * len(rows), args.k, 0)
    for name, features in feature_sets.items():
        pred = _oof_lambdamart(rows, features, target, groups, **settings)
        metrics[name] = _evaluate_utility_rankings(rows, pred, args.k)
        system_rankings[name] = _score_rankings_from_scores(rows, pred)
    nested_metrics, nested_choices, nested_rankings = _nested_residual(
        rows, settings, args.k, return_rankings=True)
    metrics["M3_nested_residual"] = nested_metrics
    system_rankings["M3_nested_residual"] = nested_rankings
    comparisons = {
        "M1_vs_frozen": {
            metric: _paired(metrics["M1_dense_fusion"], metrics["frozen_route_baseline"],
                            metric, settings["seed"])
            for metric in ("utility_ndcg_at_k", "utility_mean", "reasonable_share")},
        "M2_vs_M1": {metric: _paired(metrics["M2_official_graph"],
                                      metrics["M1_dense_fusion"], metric, settings["seed"])
                     for metric in ("utility_ndcg_at_k", "utility_mean", "reasonable_share")},
        "M3_vs_M2": {metric: _paired(metrics["M3_literature_routes"],
                                      metrics["M2_official_graph"], metric, settings["seed"])
                     for metric in ("utility_ndcg_at_k", "utility_mean", "reasonable_share")},
        "M3_vs_M1": {metric: _paired(metrics["M3_literature_routes"],
                                      metrics["M1_dense_fusion"], metric, settings["seed"])
                     for metric in ("utility_ndcg_at_k", "utility_mean", "reasonable_share")},
        "M3_nested_residual_vs_M1": {
            metric: _paired(metrics["M3_nested_residual"], metrics["M1_dense_fusion"],
                            metric, settings["seed"])
            for metric in ("utility_ndcg_at_k", "utility_mean", "reasonable_share")},
        "M3_nested_residual_vs_frozen": {
            metric: _paired(metrics["M3_nested_residual"], metrics["frozen_route_baseline"],
                            metric, settings["seed"])
            for metric in ("utility_ndcg_at_k", "utility_mean", "reasonable_share")},
    }
    summary = {name: {metric: float(np.mean([r[metric] for r in values.values()]))
                      for metric in ("utility_ndcg_at_k", "utility_mean", "reasonable_share")}
               for name, values in metrics.items()}
    sensitivity_cfg = config.params("utility_gold_sensitivity", default={}) or {}
    thresholds = tuple(float(value) for value in sensitivity_cfg.get(
        "absolute_thresholds", [4.0, 5.0]))
    top_ns = tuple(int(value) for value in sensitivity_cfg.get("per_query_top_ns", [3, 5, 8]))
    sensitivity_scores = sensitivity_comparisons = sensitivity_references = pd.DataFrame()
    disagreement_membership = disagreement_subsets = consensus_scores = pd.DataFrame()
    model_membership = model_subsets = model_consensus_scores = pd.DataFrame()
    high_disagreement = consensus_comparisons = pd.DataFrame()
    high_model_disagreement = model_consensus_comparisons = pd.DataFrame()
    robustness = {}; disagreement_summary = {}; model_disagreement_summary = {}
    disagreement_overlap = {}
    if args.heldout:
        (sensitivity_scores, sensitivity_comparisons,
         sensitivity_references, robustness) = evaluate_gold_sensitivity(
             rows, system_rankings, args.k, thresholds, top_ns, settings["seed"])
        entropy_threshold = float(sensitivity_cfg.get("consensus_entropy_threshold", 0.9))
        (disagreement_membership, disagreement_subsets,
         full_gold_rules) = analyse_rule_disagreement(
             rows, thresholds, top_ns, entropy_threshold)
        (consensus_scores, high_disagreement,
         consensus_comparisons, disagreement_summary) = (
            consensus_core_sensitivity(
                rows, system_rankings, args.k, thresholds, top_ns, settings["seed"],
                disagreement_membership, full_gold_rules))
        model_membership, model_subsets = analyse_model_disagreement(
            rows, system_rankings, args.k,
            int(sensitivity_cfg.get("model_rank_range_threshold", 6)),
            float(sensitivity_cfg.get("model_topk_entropy_threshold", 0.85)))
        (model_consensus_scores, high_model_disagreement,
         model_consensus_comparisons, model_disagreement_summary) = (
            consensus_core_sensitivity(
                rows, system_rankings, args.k, thresholds, top_ns, settings["seed"],
                model_membership, full_gold_rules))
        label_keys = set(zip(high_disagreement["query_id"].astype(str),
                             high_disagreement["comment_id"].astype(str)))
        model_keys = set(zip(high_model_disagreement["query_id"].astype(str),
                             high_model_disagreement["comment_id"].astype(str)))
        disagreement_overlap = {
            "label_rule_only": len(label_keys - model_keys),
            "model_rank_only": len(model_keys - label_keys),
            "both": len(label_keys & model_keys),
            "neither": len(rows) - len(label_keys | model_keys),
        }
    manifest = {
        "protocol": "partial-official utility M1/M2/M3 grouped OOF v1",
        "test_split_used": False, "pool_rows": len(admin), "rated_rows": len(rows),
        "missing_rows": len(missing), "complete": not missing,
        "reportable": not missing, "allow_incomplete": args.allow_incomplete,
        "label_role": "LLM utility-v2 silver; human calibration still required",
        "features": {k: list(v) for k, v in feature_sets.items()},
        "settings": settings, "nested_residual": {
            "outer_folds": settings["folds"], "inner_folds": settings["folds"],
            "candidate_budgets": [0, 1, 2], "reasonable_share_max_drop": 0.005,
            "fold_choices": nested_choices,
        }, "summary": summary, "paired": comparisons,
        "gold_sensitivity": {
            "enabled": bool(args.heldout), "k": args.k,
            "absolute_thresholds": list(thresholds), "per_query_top_ns": list(top_ns),
            "systems": list(system_rankings), "robustness": robustness,
            "disagreement_audit": {
                "primitive_rules": ["community"] +
                [f"utility_ge_{value:g}" for value in thresholds] +
                [f"utility_top_{value}" for value in top_ns],
                "entropy_threshold": float(sensitivity_cfg.get(
                    "consensus_entropy_threshold", 0.9)),
                **disagreement_summary,
                "interpretation": (
                    "Rules are correlated operational definitions, not independent raters; "
                    "subset disagreement is descriptive."),
            },
            "model_disagreement_audit": {
                "rank_range_threshold": int(sensitivity_cfg.get(
                    "model_rank_range_threshold", 6)),
                "topk_entropy_threshold": float(sensitivity_cfg.get(
                    "model_topk_entropy_threshold", 0.85)),
                **model_disagreement_summary,
                "overlap_with_label_rule_disagreement": disagreement_overlap,
                "interpretation": (
                    "This screen is label-blind and identifies candidates informative for "
                    "ranker comparison; excluding them is still a restricted-domain sensitivity."),
            },
            "claim_guard": (
                "Threshold/top-N rules are transformations of the same LLM-silver labels; "
                "cross-rule robustness is not independent human validation."),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.heldout:
        sensitivity_scores.to_csv(args.out_dir / "gold_sensitivity_scores.csv", index=False)
        sensitivity_comparisons.to_csv(
            args.out_dir / "gold_sensitivity_best_vs_runner.csv", index=False)
        sensitivity_references.to_csv(
            args.out_dir / "gold_sensitivity_reference_comparisons.csv", index=False)
        sensitivity_scores.pivot(
            index="gold_rule", columns="system", values=f"nDCG@{args.k}"
        ).to_csv(args.out_dir / "gold_sensitivity_ndcg_pivot.csv")
        pd.DataFrame([
            {"system": system, **values} for system, values in robustness.items()
        ]).sort_values(["mean_rank", "system"]).to_csv(
            args.out_dir / "gold_sensitivity_robustness.csv", index=False)
        disagreement_membership.to_csv(
            args.out_dir / "gold_rule_candidate_disagreement.csv", index=False)
        disagreement_subsets.to_csv(
            args.out_dir / "gold_rule_subset_disagreement.csv", index=False)
        high_disagreement.to_csv(
            args.out_dir / "ADMIN_high_disagreement_candidates.csv", index=False)
        consensus_scores.to_csv(
            args.out_dir / "consensus_core_scores.csv", index=False)
        consensus_comparisons.to_csv(
            args.out_dir / "consensus_core_best_vs_runner.csv", index=False)
        consensus_scores.pivot(
            index="gold_rule", columns="system", values=f"nDCG@{args.k}"
        ).to_csv(args.out_dir / "consensus_core_ndcg_pivot.csv")
        model_membership.to_csv(
            args.out_dir / "model_rank_candidate_disagreement.csv", index=False)
        model_subsets.to_csv(
            args.out_dir / "model_rank_subset_disagreement.csv", index=False)
        high_model_disagreement.to_csv(
            args.out_dir / "ADMIN_high_model_disagreement_candidates.csv", index=False)
        model_consensus_scores.to_csv(
            args.out_dir / "model_consensus_core_scores.csv", index=False)
        model_consensus_comparisons.to_csv(
            args.out_dir / "model_consensus_core_best_vs_runner.csv", index=False)
        model_consensus_scores.pivot(
            index="gold_rule", columns="system", values=f"nDCG@{args.k}"
        ).to_csv(args.out_dir / "model_consensus_core_ndcg_pivot.csv")
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
