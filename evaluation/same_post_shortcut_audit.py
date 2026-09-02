"""Validation-only audit of the same-post retrieval shortcut.

The Reddit structural gold for a query consists of replies to that same post.
This audit separates three claims:

1. the original closed-corpus structural and pooled-utility result;
2. what remains after removing every comment from the query's own post;
3. retrieval of independently judged, cross-post LLM-utility evidence.

It never converts an empty leave-one-post-out structural gold set into a score
of zero.  Such a metric is undefined and is reported as non-evaluable.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.external_fusion_utility_rerank import _read_admin
from evaluation.ir_metrics import (
    eval_full,
    graded_ndcg_at,
    hit_at,
    recall_at,
)
from evaluation.pooled_route_utility import _bare, score_routes
from data_preparation.sampling.sample_human_annotation_candidates import _load_cards
from evaluation.score_multihop_retrieval import (
    _load_pred,
    paired_bootstrap_delta,
)


def filter_same_post(
    ranked: list[str], query_post_id: str, comment_to_post: dict[str, str]
) -> list[str]:
    """Remove comments belonging to the query post, preserving rank order."""
    return [
        comment_id for comment_id in ranked
        if comment_to_post.get(str(comment_id)) != str(query_post_id)
    ]


def _gold_ids(value: object) -> list[str]:
    return [item for item in str(value).split("|") if item and item != "nan"]


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def _structural_summary(
    predictions: dict[str, list[str]],
    heldout: pd.DataFrame,
    comment_to_post: dict[str, str],
    ks: tuple[int, ...],
) -> tuple[dict, dict[str, dict[str, float]]]:
    rows = []
    per_query: dict[str, dict[str, float]] = {}
    same_share: dict[int, list[float]] = {k: [] for k in ks}
    remaining_gold = 0
    all_gold_same_post = True
    for record in heldout.itertuples(index=False):
        qid = str(record.post_id)
        ranked = predictions.get(qid, [])
        gold = set(_gold_ids(record.gold_comment_ids))
        if not gold:
            continue
        values = eval_full(ranked, gold, recall_ks=ks)
        rows.append(values)
        per_query[qid] = {key: float(value) for key, value in values.items()}
        cross_gold = {
            comment_id for comment_id in gold
            if comment_to_post.get(comment_id) != qid
        }
        remaining_gold += len(cross_gold)
        all_gold_same_post &= not cross_gold
        for k in ks:
            top = ranked[:k]
            same_share[k].append(
                sum(comment_to_post.get(comment_id) == qid for comment_id in top)
                / k
            )
    keys = list(rows[0]) if rows else []
    return {
        "queries": len(rows),
        "metrics": {key: _mean([row[key] for row in rows]) for key in keys},
        "same_post_share": {f"@{k}": _mean(values)
                            for k, values in same_share.items()},
        "leave_one_post_out_structural": {
            "evaluable_queries": 0 if all_gold_same_post else None,
            "remaining_gold_comments": remaining_gold,
            "metric_status": (
                "undefined_all_structural_gold_is_same_post"
                if all_gold_same_post else "requires_cross_post_structural_gold"
            ),
        },
    }, per_query


def _cross_post_utility(
    predictions: dict[str, list[str]],
    query_ids: set[str],
    gains: dict[tuple[str, str], float],
    pool: dict[str, set[str]],
    comment_to_post: dict[str, str],
    *,
    k: int,
    reasonable_threshold: float,
) -> tuple[dict, dict[str, dict[str, float]]]:
    per_query: dict[str, dict[str, float]] = {}
    pool_counts = []
    for qid in sorted(query_ids):
        cross_ids = {
            comment_id for comment_id in pool.get(qid, set())
            if comment_to_post.get(comment_id) != qid
        }
        if not cross_ids:
            continue
        query_gains = {
            comment_id: gains[(qid, comment_id)] for comment_id in cross_ids
        }
        reasonable_gold = {
            comment_id for comment_id, value in query_gains.items()
            if value >= reasonable_threshold
        }
        raw = predictions.get(qid, [])
        lopo = filter_same_post(raw, qid, comment_to_post)
        projected = [comment_id for comment_id in lopo
                     if comment_id in query_gains]
        per_query[qid] = {
            # Full-list metrics retain rank displacement; unjudged candidates
            # receive zero gain, as in a conventional pooled TREC evaluation.
            f"raw_cross_utility_ndcg@{k}": graded_ndcg_at(raw, query_gains, k),
            f"lopo_cross_utility_ndcg@{k}": graded_ndcg_at(lopo, query_gains, k),
            # Projection isolates ordering among judged cross-post candidates.
            f"projected_cross_utility_ndcg@{k}": graded_ndcg_at(
                projected, query_gains, k),
            f"raw_cross_reasonable_recall@{k}": recall_at(
                raw, reasonable_gold, k),
            f"lopo_cross_reasonable_recall@{k}": recall_at(
                lopo, reasonable_gold, k),
            f"raw_cross_reasonable_hit@{k}": hit_at(raw, reasonable_gold, k),
            f"lopo_cross_reasonable_hit@{k}": hit_at(lopo, reasonable_gold, k),
        }
        pool_counts.append(len(cross_ids))
    metric_names = list(next(iter(per_query.values()))) if per_query else []
    return {
        "queries_with_cross_post_judgments": len(per_query),
        "queries_without_cross_post_judgments": len(query_ids) - len(per_query),
        "cross_post_judged_candidates": sum(pool_counts),
        "mean_cross_post_judged_candidates_per_query": _mean(pool_counts),
        "metrics": {
            metric: _mean([row[metric] for row in per_query.values()])
            for metric in metric_names
        },
        "pooling_note": (
            "Full-list metrics treat unjudged candidates as zero gain; projected "
            "nDCG isolates ordering among judged cross-post candidates."
        ),
    }, per_query


def run_audit(
    corpus_map: Path,
    heldout_path: Path,
    admin_pool: Path,
    study_db: Path,
    prediction_paths: dict[str, Path],
    *,
    k: int = 8,
    reasonable_threshold: float = 4.0,
) -> dict:
    guarded_paths = [corpus_map, heldout_path, admin_pool, study_db,
                     *prediction_paths.values()]
    if any("test" in path.name.lower() for path in guarded_paths):
        raise ValueError("frozen test artifacts are forbidden")

    corpus = pd.read_csv(corpus_map, dtype=str)
    heldout = pd.read_csv(heldout_path, dtype=str)
    comment_to_post = dict(zip(corpus["comment_id"], corpus["post_id"]))
    query_ids = set(heldout["post_id"].astype(str))

    admin = _read_admin(admin_pool)
    cards = _load_cards(study_db, heldout_path)
    gains = {
        (_bare(row["query_id"]), str(row["comment_id"])): float(row["utility"])
        for row in cards
    }
    pool: dict[str, set[str]] = defaultdict(set)
    for row in admin:
        qid = _bare(row.get("canonical_query_id") or row["query_id"])
        pool[qid].add(str(row["comment_id"]))
    missing = [
        (qid, comment_id) for qid, comments in pool.items()
        for comment_id in comments if (qid, comment_id) not in gains
    ]
    if missing:
        raise RuntimeError(f"utility pool has {len(missing)} unrated candidates")

    systems = {}
    per_system_cross = {}
    for name, path in prediction_paths.items():
        predictions = {_bare(qid): ranking
                       for qid, ranking in _load_pred(path).items()}
        structural, _ = _structural_summary(
            predictions, heldout, comment_to_post, (5, 10, 20))
        cross, per_cross = _cross_post_utility(
            predictions, query_ids & set(pool), gains, pool, comment_to_post,
            k=k, reasonable_threshold=reasonable_threshold)
        systems[name] = {
            "prediction_path": str(path),
            "closed_corpus_structural": structural,
            "cross_post_llm_utility": cross,
        }
        raw_metric = f"raw_cross_utility_ndcg@{k}"
        lopo_metric = f"lopo_cross_utility_ndcg@{k}"
        systems[name]["same_post_removal_effect"] = {
            "metric": lopo_metric,
            "comparison": "leave_one_post_out_minus_raw",
            **(paired_bootstrap_delta(
                {qid: row[lopo_metric] for qid, row in per_cross.items()},
                {qid: row[raw_metric] for qid, row in per_cross.items()},
            ) or {}),
            "interpretation": (
                "Mechanical rank-space released by removing same-post comments; "
                "not evidence that the retrieval model itself improved."
            ),
        }
        per_system_cross[name] = per_cross

    closed_utility = score_routes(
        admin_pool, study_db, heldout_path, prediction_paths,
        k=k, reasonable_threshold=reasonable_threshold)
    comparisons = []
    names = list(prediction_paths)
    comparison_metrics = (
        f"lopo_cross_utility_ndcg@{k}",
        f"projected_cross_utility_ndcg@{k}",
        f"lopo_cross_reasonable_recall@{k}",
        f"lopo_cross_reasonable_hit@{k}",
    )
    for metric in comparison_metrics:
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                delta = paired_bootstrap_delta(
                    {qid: row[metric]
                     for qid, row in per_system_cross[left].items()},
                    {qid: row[metric]
                     for qid, row in per_system_cross[right].items()},
                )
                comparisons.append({
                    "left": left,
                    "right": right,
                    "metric": metric,
                    **(delta or {}),
                })

    same_pool = sum(
        comment_to_post.get(comment_id) == qid
        for qid, comments in pool.items() for comment_id in comments
    )
    return {
        "protocol": "same-post shortcut audit v1",
        "test_split_used": False,
        "corpus_map": str(corpus_map),
        "heldout": str(heldout_path),
        "validation_queries": len(query_ids),
        "utility_pool_queries": len(pool),
        "utility_pool_candidates": sum(map(len, pool.values())),
        "utility_pool_same_post_candidates": same_pool,
        "utility_pool_cross_post_candidates": sum(map(len, pool.values())) - same_pool,
        "reasonable_threshold": reasonable_threshold,
        "k": k,
        "closed_corpus_pooled_utility": closed_utility["systems"],
        "systems": systems,
        "paired_lopo_cross_post": comparisons,
        "claim_guard": (
            "Community structural gold has no leave-one-post-out positives. "
            "Cross-post LLM utility is silver and pool-incomplete; human calibration "
            "is still required."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-map", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--admin-pool", type=Path, required=True)
    parser.add_argument("--study-db", type=Path, required=True)
    parser.add_argument("--pred", action="append", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--reasonable-threshold", type=float, default=4.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predictions = {}
    for item in args.pred:
        name, sep, path = item.partition("=")
        if not sep:
            raise ValueError(f"invalid --pred value: {item}")
        predictions[name] = Path(path)
    result = run_audit(
        args.corpus_map, args.heldout, args.admin_pool, args.study_db,
        predictions, k=args.k,
        reasonable_threshold=args.reasonable_threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
