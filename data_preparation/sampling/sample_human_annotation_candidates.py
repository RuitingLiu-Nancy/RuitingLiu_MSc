#!/usr/bin/env python3
"""Stratified candidate sampler for human evidence annotation.

This script reads the deep evidence-card pool from study_platform's SQLite DB
(`ev_cards` + LLM `ev_card_ratings`) and proposes a per-query subset for human
annotation. The goal is diagnostic supervision rather than random population
estimation: keep strong baseline evidence, route-exclusive evidence, community
gold, borderline examples, and safe negatives so a reranker can learn useful
top-8 support selection.

Example:
  python data_preparation/sampling/sample_human_annotation_candidates.py \
    --study-db ../study_platform/backend/data/study.db \
    --heldout out/expanded_graph_rebuild/heldout_validation.csv \
    --out-dir out/human_annotation_sampling/deep_llm_pilot
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(10_000_000)

DIMS = ("relevance", "usefulness", "actionability", "novelty", "safety")
WEIGHTS = {
    "relevance": 0.30,
    "usefulness": 0.30,
    "actionability": 0.15,
    "novelty": 0.15,
    "safety": 0.10,
}

SLOTS = (
    ("community_gold", 2),
    ("dense_or_fusion_high", 2),
    ("graph_only_high", 2),
    ("graph_only_borderline", 2),
    ("fusion_or_bm25_exclusive", 1),
    ("llm_high_not_community_gold", 1),
    ("safe_negative_control", 1),
    ("boundary_or_uncertain", 1),
)


def _candidate_utility(ratings: dict) -> float | None:
    if any(ratings.get(k) is None for k in DIMS):
        return None
    value = sum(float(ratings[k]) * WEIGHTS[k] for k in DIMS)
    if float(ratings["safety"]) <= 2:
        value = min(value, 2.0)
    return value


def _load_community_gold_ids(path: Path | None) -> dict[str, set[str]]:
    if not path:
        return {}
    out: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            qid = r.get("post_id") or r.get("query_id") or ""
            ids = (r.get("gold_comment_ids") or "").replace(",", "|").split("|")
            out[qid] = {x.strip() for x in ids if x.strip()}
    return out


def _load_cards(db_path: Path, heldout: Path | None) -> list[dict]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    gold_by_q = _load_community_gold_ids(heldout)
    ratings_by_card: dict[str, dict] = {}
    raters_by_card: dict[str, set[str]] = defaultdict(set)
    for r in con.execute("SELECT card_id,pseudonym,ratings_json FROM ev_card_ratings"):
        payload = json.loads(r["ratings_json"] or "{}")
        if not all(k in payload for k in DIMS):
            continue
        ratings_by_card[r["card_id"]] = payload
        raters_by_card[r["card_id"]].add(r["pseudonym"])

    rows: list[dict] = []
    sql = (
        "SELECT c.id AS card_id,c.query_id,c.comment_id,c.snippet,"
        "c.facets_json,c.from_arms_json,q.text AS query_text "
        "FROM ev_cards c JOIN queries q ON q.id=c.query_id"
    )
    for r in con.execute(sql):
        ratings = ratings_by_card.get(r["card_id"])
        if not ratings:
            continue
        utility = _candidate_utility(ratings)
        if utility is None:
            continue
        arms = json.loads(r["from_arms_json"] or "{}")
        facets = json.loads(r["facets_json"] or "{}")
        arm_set = set(arms)
        rows.append({
            "query_id": r["query_id"],
            "comment_id": str(r["comment_id"]),
            "card_id": r["card_id"],
            "query_text": r["query_text"] or "",
            "snippet": r["snippet"] or "",
            "from_arms": "|".join(sorted(arm_set)),
            "semantic_rank": int(arms.get("semantic", 0) or 0),
            "fusion_dense_bm25_rank": int(arms.get("fusion_dense_bm25", 0) or 0),
            "graph_rank": int(arms.get("graph", 0) or 0),
            "n_sources": len(arm_set),
            "graph_only": int(arm_set == {"graph"}),
            "semantic_only": int(arm_set == {"semantic"}),
            "fusion_only": int(arm_set == {"fusion_dense_bm25"}),
            "community_gold": int(str(r["comment_id"]) in gold_by_q.get(r["query_id"], set())),
            "utility": round(utility, 4),
            "label_relevance": ratings.get("relevance"),
            "label_usefulness": ratings.get("usefulness"),
            "label_actionability": ratings.get("actionability"),
            "label_novelty": ratings.get("novelty"),
            "label_safety": ratings.get("safety"),
            "n_llm_raters": len(raters_by_card[r["card_id"]]),
            "domain_count": len(facets.get("domains") or []),
            "ef_count": len(facets.get("ef") or []),
            "support_count": len(facets.get("support") or []),
            "entity_count": len(facets.get("entities") or []),
        })
    return rows


def _best_rank(row: dict) -> int:
    ranks = [row[k] for k in ("semantic_rank", "fusion_dense_bm25_rank", "graph_rank") if row[k]]
    return min(ranks) if ranks else 9999


def _slot_candidates(rows: list[dict], slot: str) -> list[dict]:
    if slot == "community_gold":
        return [r for r in rows if r["community_gold"]]
    if slot == "dense_or_fusion_high":
        return [
            r for r in rows
            if r["utility"] >= 4 and (r["semantic_rank"] or r["fusion_dense_bm25_rank"])
        ]
    if slot == "graph_only_high":
        return [r for r in rows if r["graph_only"] and r["utility"] >= 4]
    if slot == "graph_only_borderline":
        return [r for r in rows if r["graph_only"] and 3.25 <= r["utility"] < 4.25]
    if slot == "fusion_or_bm25_exclusive":
        return [r for r in rows if r["fusion_only"] and r["utility"] >= 3.5]
    if slot == "llm_high_not_community_gold":
        return [r for r in rows if r["utility"] >= 5 and not r["community_gold"]]
    if slot == "safe_negative_control":
        return [r for r in rows if r["utility"] < 3.25 and float(r["label_safety"]) >= 4]
    if slot == "boundary_or_uncertain":
        return [r for r in rows if 3.75 <= r["utility"] < 4.25]
    raise ValueError(slot)


def _sort_for_slot(rows: list[dict], slot: str) -> list[dict]:
    if slot in {"safe_negative_control", "graph_only_borderline", "boundary_or_uncertain"}:
        return sorted(rows, key=lambda r: (abs(r["utility"] - 4.0), _best_rank(r), r["card_id"]))
    return sorted(rows, key=lambda r: (-r["utility"], _best_rank(r), r["card_id"]))


def sample(rows: list[dict], per_query: int) -> list[dict]:
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_q[r["query_id"]].append(r)

    selected: list[dict] = []
    for qid in sorted(by_q):
        qrows = by_q[qid]
        picked: dict[str, dict] = {}

        def add(row: dict, slot: str, priority: int) -> bool:
            if row["card_id"] in picked:
                return False
            out = dict(row)
            out["sample_stratum"] = slot
            out["sample_priority"] = priority
            picked[row["card_id"]] = out
            return True

        for slot, quota in SLOTS:
            priority = 0
            for cand in _sort_for_slot(_slot_candidates(qrows, slot), slot):
                if len([r for r in picked.values() if r["sample_stratum"] == slot]) >= quota:
                    break
                if add(cand, slot, priority):
                    priority += 1

        # Fill missing slots with a utility/route-diverse fallback. This keeps
        # every query close to per_query even when a rare slot is unavailable.
        fallback = sorted(qrows, key=lambda r: (-r["utility"], r["n_sources"], _best_rank(r), r["card_id"]))
        for cand in fallback:
            if len(picked) >= per_query:
                break
            add(cand, "fallback_utility_diverse", len(picked))

        selected.extend(sorted(picked.values(), key=lambda r: (r["sample_stratum"], r["sample_priority"], r["card_id"]))[:per_query])
    return selected


def _sampling_manifest(rows: list[dict], selected: list[dict]) -> dict:
    by_q = defaultdict(int)
    for r in selected:
        by_q[r["query_id"]] += 1
    return {
        "pool_cards": len(rows),
        "pool_queries": len({r["query_id"] for r in rows}),
        "selected_cards": len(selected),
        "selected_queries": len(by_q),
        "selected_per_query": {
            "min": min(by_q.values()) if by_q else 0,
            "max": max(by_q.values()) if by_q else 0,
            "mean": round(sum(by_q.values()) / len(by_q), 2) if by_q else 0,
        },
        "strata": dict(Counter(r["sample_stratum"] for r in selected)),
        "arms": dict(Counter(r["from_arms"] for r in selected)),
        "community_gold_selected": sum(r["community_gold"] for r in selected),
        "utility_ge4_selected": sum(r["utility"] >= 4 for r in selected),
        "utility_ge5_selected": sum(r["utility"] >= 5 for r in selected),
        "graph_only_selected": sum(r["graph_only"] for r in selected),
        "graph_only_utility_ge4_selected": sum(r["graph_only"] and r["utility"] >= 4 for r in selected),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-db", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, default=None,
                    help="heldout CSV with post_id/query_id + gold_comment_ids")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--per-query", type=int, default=12)
    args = ap.parse_args()

    rows = _load_cards(args.study_db, args.heldout)
    selected = sample(rows, args.per_query)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "human_annotation_candidates.csv"
    fields = [
        "query_id", "comment_id", "card_id", "sample_stratum", "sample_priority",
        "utility", "community_gold", "from_arms", "semantic_rank",
        "fusion_dense_bm25_rank", "graph_rank", "n_sources", "graph_only",
        "semantic_only", "fusion_only", "label_relevance", "label_usefulness",
        "label_actionability", "label_novelty", "label_safety", "n_llm_raters",
        "domain_count", "ef_count", "support_count", "entity_count",
        "query_text", "snippet",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in selected)

    manifest = _sampling_manifest(rows, selected)
    manifest.update({
        "out_csv": str(out_csv),
        "study_db": str(args.study_db),
        "heldout": str(args.heldout) if args.heldout else None,
        "per_query_requested": args.per_query,
        "slots": dict(SLOTS),
        "note": (
            "Diagnostic stratified sampling for human supervision; not an "
            "unbiased prevalence estimate of useful comments."
        ),
    })
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
