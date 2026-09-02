#!/usr/bin/env python3
"""Create mixed validation/test query splits from original and expanded audits.

This script operates on post-level role-audit CSVs. It keeps only safe
query-candidate posts, records the source corpus, then draws balanced validation
and test query sets. It also marks a smaller human-priority subset in each
split, leaving the remainder for LLM-first judging.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_exclude(paths: list[Path]) -> set[str]:
    out: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                pid = (r.get("query_id") or r.get("post_id") or "").strip()
                if pid:
                    out.add(pid)
    return out


def _load_audit(path: Path, source: str, exclude: set[str]) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pid = (r.get("post_id") or "").strip()
            if not pid or pid in exclude:
                continue
            if r.get("use_bucket") != "query_candidate":
                continue
            if str(r.get("drug_related", "0")) != "0":
                continue
            rows.append({
                "query_id": pid,
                "post_id": pid,
                "source": source,
                "scenario": r.get("scenario") or "UNKNOWN",
                "tier": r.get("tier") or "UNKNOWN",
                "created_year": r.get("created_year") or "",
                "post_role": r.get("post_role") or "",
                "role_confidence": r.get("role_confidence") or "",
                "role_reason": r.get("role_reason") or "",
                "drug_related": r.get("drug_related") or "0",
                "drug_match": r.get("drug_match") or "",
                "query_text": r.get("post_text") or r.get("query_text") or "",
            })
    return rows


def _balance_key(row: dict, mode: str) -> str:
    parts = []
    if "scenario" in mode:
        parts.append(row.get("scenario", "UNKNOWN"))
    if "tier" in mode:
        parts.append(row.get("tier", "UNKNOWN"))
    if "source" in mode:
        parts.append(row.get("source", "UNKNOWN"))
    return "::".join(parts) or "ALL"


def _balanced_take(rows: list[dict], n: int, rng: random.Random,
                   mode: str) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[_balance_key(row, mode)].append(row)
    for key in buckets:
        rng.shuffle(buckets[key])
    selected: list[dict] = []
    seen: set[str] = set()
    keys = sorted(buckets)
    while len(selected) < n:
        moved = False
        for key in keys:
            bucket = buckets[key]
            while bucket and bucket[-1]["query_id"] in seen:
                bucket.pop()
            if bucket and len(selected) < n:
                row = bucket.pop()
                selected.append(row)
                seen.add(row["query_id"])
                moved = True
        if not moved:
            break
    return selected


def _assign_human_priority(rows: list[dict], n: int, rng: random.Random,
                           mode: str) -> None:
    chosen = {r["query_id"] for r in _balanced_take(rows, min(n, len(rows)), rng, mode)}
    for row in rows:
        row["judge_priority"] = "human" if row["query_id"] in chosen else "llm_first"


def _write(path: Path, split: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split", "query_id", "post_id", "source", "scenario", "tier",
        "created_year", "post_role", "role_confidence", "role_reason",
        "drug_related", "drug_match", "judge_priority", "query_text",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = dict(row)
            out["split"] = split
            w.writerow({k: out.get(k, "") for k in fields})


def _counts(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "source": dict(Counter(r["source"] for r in rows)),
        "scenario": dict(Counter(r["scenario"] for r in rows)),
        "tier": dict(Counter(r["tier"] for r in rows)),
        "post_role": dict(Counter(r["post_role"] for r in rows)),
        "judge_priority": dict(Counter(r.get("judge_priority", "") for r in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original-audit", type=Path, required=True)
    ap.add_argument("--expanded-audit", type=Path, required=True)
    ap.add_argument("--exclude", nargs="*", type=Path, default=[])
    ap.add_argument("--validation-n", type=int, default=400)
    ap.add_argument("--test-n", type=int, default=400)
    ap.add_argument("--human-validation-n", type=int, default=100)
    ap.add_argument("--human-test-n", type=int, default=100)
    ap.add_argument("--balance", default="scenario_tier_source",
                    choices=["scenario", "scenario_tier", "scenario_source",
                             "scenario_tier_source"])
    ap.add_argument("--seed", type=int, default=20260703)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("out/mixed_query_splits_800"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    exclude = _load_exclude(args.exclude)
    rows = []
    rows.extend(_load_audit(args.original_audit, "original_1105", exclude))
    rows.extend(_load_audit(args.expanded_audit, "expanded_4x", exclude))

    pool = list(rows)
    validation = _balanced_take(pool, min(args.validation_n, len(pool)), rng,
                                args.balance)
    validation_ids = {r["query_id"] for r in validation}
    remaining = [r for r in pool if r["query_id"] not in validation_ids]
    test = _balanced_take(remaining, min(args.test_n, len(remaining)), rng,
                          args.balance)
    test_ids = {r["query_id"] for r in test}
    reserve = [r for r in remaining if r["query_id"] not in test_ids]

    _assign_human_priority(validation, args.human_validation_n, rng, args.balance)
    _assign_human_priority(test, args.human_test_n, rng, args.balance)
    for row in reserve:
        row["judge_priority"] = "reserve"

    _write(args.out_dir / "validation_queries.csv", "validation", validation)
    _write(args.out_dir / "test_queries.csv", "test", test)
    _write(args.out_dir / "reserve_queries.csv", "reserve", reserve)
    all_rows = validation + test + reserve
    _write(args.out_dir / "all_mixed_query_candidates.csv", "mixed", all_rows)

    manifest = {
        "original_audit": str(args.original_audit),
        "original_audit_sha256": _sha256(args.original_audit),
        "expanded_audit": str(args.expanded_audit),
        "expanded_audit_sha256": _sha256(args.expanded_audit),
        "exclude_files": {
            str(path): _sha256(path)
            for path in args.exclude
            if path.exists()
        },
        "excluded_ids": len(exclude),
        "eligible_pool": _counts(pool),
        "balance": args.balance,
        "seed": args.seed,
        "validation": _counts(validation),
        "test": _counts(test),
        "reserve": _counts(reserve),
        "outputs": {
            "validation": "validation_queries.csv",
            "test": "test_queries.csv",
            "reserve": "reserve_queries.csv",
            "all_candidates": "all_mixed_query_candidates.csv",
        },
        "note": ("Use validation for tuning and reranker training. Keep test for "
                 "final evaluation; test LLM scores are evaluation labels, not "
                 "training labels."),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
