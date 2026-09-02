#!/usr/bin/env python3
"""STEP 1 of the data-driven pipeline: a BASE sample using ONLY minimal,
annotation-free filters. No scenario, no clustering, no embeddings, no priors.

This is deliberately the most neutral draw possible, so that the structure we
later find via clustering is discovered FROM the data, not imposed by our
sampling choices. Everything downstream (clustering -> dimensions -> balanced
re-sample) builds on this.

Exact, explicit filters (the ONLY criteria used at this step):
  - subreddit in --subreddits (default: ADHD)
  - created_utc year in [--year-min, --year-max]  (default 2023-2025)
  - is_self == True                               (a real text post)
  - selftext not deleted/removed/empty
  - len(selftext) >= --min-post-chars             (default 300)
  - >= --min-comments qualifying top-level comments (default 1; keeps shallow too)
  - qualifying comment = top-level (parent_id == link_id), not deleted/bot/
    url-only, len(body) >= --min-comment-chars (default 150)

NOTE: we do NOT exclude already-annotated posts here. Step 1 is meant to be a
fully NEUTRAL draw over ALL qualifying recent posts, so the structure found by
clustering reflects the true population, not "the part we hadn't labelled yet".
Cost-saving (reusing existing labels for any post already annotated) is deferred
to the FINAL balanced-sampling step, not imposed on the discovery sample.
(--exclude is still available but defaults to none.)

Sampling: if the qualifying pool exceeds --n, take a UNIFORM RANDOM sample of
--n posts (seed fixed). No stratification of any kind here — that is the whole
point of Step 1.

Output: base_sample_posts.jsonl  (post_id, year, n_comments, post_text, comments)
        base_sample_stats.json   (counts, year/comment-depth distributions)

Run on your machine (streams the raw dump; not runnable in the sandbox).

Usage:
  python step1_base_sample.py \
    --dumps-dir ../adhd_data_reddit/subreddits25 \
    --subreddits ADHD --year-min 2023 --year-max 2025 \
    --min-post-chars 300 --min-comment-chars 150 --min-comments 1 \
    --n 5000 --seed 20260610 \
    --exclude out/annotation_input_merged_v2.csv \
    --out out/step1/base_sample_posts.jsonl \
    --stats out/step1/base_sample_stats.json
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2_147_483_647))

BOT_AUTHORS = {"AutoModerator", "[deleted]", "[removed]", "B0tRank", "RepostSleuthBot"}
DELETED = {"[deleted]", "[removed]", "", "nan", None}
URL_ONLY = re.compile(r"^\s*https?://\S+\s*$")


def stream_zst(path: Path):
    proc = subprocess.Popen(["zstd", "-dc", "--long=31", str(path)],
                            stdout=subprocess.PIPE, bufsize=1 << 20)
    try:
        for line in proc.stdout:
            try:
                yield json.loads(line)
            except Exception:
                continue
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()


def year_of(o):
    try:
        return dt.datetime.utcfromtimestamp(int(o.get("created_utc"))).year
    except Exception:
        return None


def load_exclude(path):
    ids = set()
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                pid = (r.get("post_id") or "").strip()
                if pid:
                    ids.add(pid)
    return ids


def run(args):
    exclude = load_exclude(args.exclude)
    subs = args.subreddits.split(",") if args.subreddits else None
    dumps = Path(args.dumps_dir)
    print(f"[step1] excluding {len(exclude)} already-annotated post_ids", flush=True)

    qualifying = []  # full post records that pass ALL base filters
    for sub_path in sorted(dumps.glob("*_submissions.zst")):
        name = sub_path.name[:-len("_submissions.zst")]
        if subs and name not in subs:
            continue
        cmt_path = dumps / f"{name}_comments.zst"
        if not cmt_path.exists():
            continue
        idx = {}
        for s in stream_zst(sub_path):
            y = year_of(s)
            if y is None or y < args.year_min or y > args.year_max:
                continue
            if not s.get("is_self"):
                continue
            txt = s.get("selftext") or ""
            if txt in DELETED or len(txt) < args.min_post_chars:
                continue
            pid = s.get("id", "")
            if not pid or pid in exclude:
                continue
            idx["t3_" + pid] = {"post_id": pid, "year": y, "subreddit": name,
                                "post_text": txt, "comments": []}
        for c in stream_zst(cmt_path):
            link = c.get("link_id")
            if link not in idx or c.get("parent_id") != link:
                continue
            body = c.get("body") or ""
            if body in DELETED or len(body) < args.min_comment_chars:
                continue
            if c.get("author") in BOT_AUTHORS or URL_ONLY.match(body):
                continue
            idx[link]["comments"].append(
                {"comment_id": c.get("id", ""), "body": body,
                 "score": int(c.get("score", 0) or 0)})
        for rec in idx.values():
            if len(rec["comments"]) >= args.min_comments:
                rec["n_comments"] = len(rec["comments"])
                qualifying.append(rec)
        print(f"[step1] {name}: {len(idx)} window posts, "
              f"{len(qualifying)} qualifying so far", flush=True)

    n_pool = len(qualifying)
    rng = random.Random(args.seed)
    if args.n and n_pool > args.n:
        sample = rng.sample(qualifying, args.n)
    else:
        sample = qualifying
    print(f"[step1] qualifying pool {n_pool}; drew {len(sample)} "
          f"(uniform random, seed={args.seed})", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in sample:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    depths = sorted(r["n_comments"] for r in sample)
    years = Counter(r["year"] for r in sample)
    stats = {
        "filters": {"subreddits": args.subreddits, "year_min": args.year_min,
                    "year_max": args.year_max, "min_post_chars": args.min_post_chars,
                    "min_comment_chars": args.min_comment_chars,
                    "min_comments": args.min_comments},
        "qualifying_pool": n_pool, "sampled": len(sample), "seed": args.seed,
        "year_distribution": dict(sorted(years.items())),
        "comment_depth": {
            "median": depths[len(depths) // 2] if depths else 0,
            "mean": round(sum(depths) / max(len(depths), 1), 1),
            "min": depths[0] if depths else 0, "max": depths[-1] if depths else 0,
            "ge4": sum(1 for d in depths if d >= 4),
            "ge16": sum(1 for d in depths if d >= 16)},
        "total_comments": sum(depths),
    }
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[step1] saved -> {out}\n[step1] stats -> {args.stats}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps-dir", type=Path, default=Path("../adhd_data_reddit/subreddits25"))
    ap.add_argument("--subreddits", default="ADHD")
    ap.add_argument("--year-min", type=int, default=2023)
    ap.add_argument("--year-max", type=int, default=2025)
    ap.add_argument("--min-post-chars", type=int, default=300)
    ap.add_argument("--min-comment-chars", type=int, default=150)
    ap.add_argument("--min-comments", type=int, default=1)
    ap.add_argument("--n", type=int, default=5000, help="uniform random sample size (0 = keep all)")
    ap.add_argument("--seed", type=int, default=20260610)
    ap.add_argument("--exclude", type=Path, default=None,
                    help="optional CSV of post_ids to exclude; default NONE "
                         "(Step 1 is a neutral draw over all qualifying posts)")
    ap.add_argument("--out", type=Path, default=Path("out/step1/base_sample_posts.jsonl"))
    ap.add_argument("--stats", type=Path, default=Path("out/step1/base_sample_stats.json"))
    a = ap.parse_args()
    run(a)


if __name__ == "__main__":
    main()
