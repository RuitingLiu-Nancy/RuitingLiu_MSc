#!/usr/bin/env python3
"""Stratified RE-SAMPLE: depth tier x scenario balanced sample from the raw dump.

Replaces the four ad-hoc legacy batches (main / shallow-hub / deep-hub / rare-
supplement) with ONE coherent design:

  - time window 2023-2025 (recent, contemporary ADHD discourse)
  - depth tiers by # qualifying top-level comments:
        shallow = 1-3, mid = 4-15, deep = >=16
  - 13 schema scenarios (general_unspecified excluded as a sampling target)
  - per (tier x scenario) QUOTA: --shallow-q / --mid-q / --deep-q
  - within a cell: prefer higher comment-count, then higher top-comment score
  - deep posts keep ALL qualifying comments (depth is the point); shallow/mid keep
    up to --max-per-post highest-score comments
  - excludes post_ids already annotated (so it can run alongside the legacy graph
    as a clean comparison set)

Scenario label = few-shot embedding prototype classifier (reuse the 2338
annotated posts as reference), same as classify_scenarios.py.

Run on your machine (streams the 1.7G dump; sentence-transformers needed).

Usage:
  python resample_stratified.py \
    --dumps-dir ../adhd_data_reddit/subreddits25 \
    --subreddits ADHD --year-min 2023 --year-max 2025 \
    --shallow-q 40 --mid-q 25 --deep-q 20 --max-per-post 8 \
    --backend bert \
    --out out/resample/resample_input_raw.csv \
    --manifest out/resample/resample_manifest.json
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
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

csv.field_size_limit(min(sys.maxsize, 2_147_483_647))

BOT_AUTHORS = {"AutoModerator", "[deleted]", "[removed]", "B0tRank", "RepostSleuthBot"}
DELETED = {"[deleted]", "[removed]", "", "nan", None}
URL_ONLY = re.compile(r"^\s*https?://\S+\s*$")
SEP = "|"
OUT_COLUMNS = ["annotation_id", "post_id", "comment_id", "target_text",
               "post_context", "subreddit", "comment_score", "comment_chars",
               "post_chars", "tier", "scenario", "created_year"]


def seed_runtime(seed: int) -> None:
    """Seed libraries used by the embedding backend.

    The sampler itself is quota/rank based rather than stochastic.  The seed
    freezes any model-runtime randomness in sentence-transformer inference and
    is therefore recorded even though it does not alter the ranking rule.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


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


def tier_of(n: int) -> str:
    if n >= 16:
        return "deep"
    if n >= 4:
        return "mid"
    if n >= 1:
        return "shallow"
    return "none"


def primary_scenario(s):
    parts = [p.strip() for p in str(s or "").split(SEP) if p.strip()]
    return parts[0] if parts else "general_unspecified"


def load_exclude(post_jsonl: Path) -> set[str]:
    ids = set()
    if post_jsonl and Path(post_jsonl).exists():
        for line in Path(post_jsonl).read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(str(json.loads(line).get("post_id", "")).strip())
    return ids


def build_classifier(
    text_csv, post_jsonl, backend_name, model, model_revision,
    embedding_batch_size,
):
    df = pd.read_csv(text_csv, dtype=str, keep_default_na=False)
    ptext = {}
    for _, r in df.iterrows():
        pid = str(r["post_id"]).strip()
        if pid and pid not in ptext:
            ptext[pid] = str(r.get("post_context", "")).strip()
    lab = {}
    for line in Path(post_jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            pid = str(o.get("post_id", "")).strip()
            if pid:
                lab[pid] = primary_scenario(o.get("scenarios", ""))
    ref = [(ptext[pid], lab[pid]) for pid in lab if pid in ptext and ptext[pid]]
    scen = sorted({l for _, l in ref if l != "general_unspecified"})  # targets only

    if backend_name == "bert":
        from sentence_transformers import SentenceTransformer
        import numpy as np
        m = SentenceTransformer(model, revision=model_revision)
        ref_emb = np.asarray(m.encode([t for t, _ in ref], normalize_embeddings=True,
                                      batch_size=embedding_batch_size,
                                      show_progress_bar=True))
        protos = {s: ref_emb[[i for i, (_, l) in enumerate(ref) if l == s]].mean(0)
                  for s in scen}
        P = np.stack([protos[s] for s in scen])
        def classify(texts):
            e = np.asarray(m.encode(list(texts), normalize_embeddings=True,
                                    batch_size=embedding_batch_size,
                                    show_progress_bar=True))
            return [scen[i] for i in (e @ P.T).argmax(1)]
        return classify, scen
    else:
        import math
        WORD = re.compile(r"[a-z]{3,}")
        def toks(t): return WORD.findall(str(t).lower())
        dfc = defaultdict(int); docs = [toks(t) for t, _ in ref]; N = len(docs)
        for d in docs:
            for w in set(d):
                dfc[w] += 1
        idf = {w: math.log((N + 1) / (n + 1)) + 1 for w, n in dfc.items()}
        def vec(t):
            tf = Counter(toks(t)); v = {w: (1 + math.log(c)) * idf.get(w, 0) for w, c in tf.items()}
            nr = math.sqrt(sum(x * x for x in v.values())) or 1
            return {w: x / nr for w, x in v.items()}
        protos = {}
        for s in scen:
            agg = defaultdict(float); c = 0
            for (t, l) in ref:
                if l == s:
                    for w, x in vec(t).items():
                        agg[w] += x
                    c += 1
            protos[s] = {w: x / c for w, x in agg.items()} if c else {}
        def cos(a, b):
            if len(a) > len(b): a, b = b, a
            return sum(x * b.get(w, 0) for w, x in a.items())
        def classify(texts):
            return [max(scen, key=lambda s: cos(vec(t), protos[s])) for t in texts]
        return classify, scen


def run(args):
    seed_runtime(args.seed)
    exclude_path = Path(args.exclude_jsonl) if args.exclude_jsonl else Path(args.post_jsonl)
    exclude = load_exclude(exclude_path)
    print(f"[resample] excluding {len(exclude)} post_ids from {exclude_path}", flush=True)
    classify, scen_targets = build_classifier(
        args.text_csv,
        args.post_jsonl,
        args.backend,
        args.model,
        args.model_revision,
        args.embedding_batch_size,
    )
    subs = args.subreddits.split(",") if args.subreddits else None
    dumps = Path(args.dumps_dir)
    quota = {"shallow": args.shallow_q, "mid": args.mid_q, "deep": args.deep_q}

    # 1) collect window posts with text + their qualifying comments
    posts = {}  # post_id -> {text, year, sub, comments:[...]}
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
            idx["t3_" + pid] = {"pid": pid, "text": txt, "year": y,
                                "sub": name, "comments": []}
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
            rec["tier"] = tier_of(len(rec["comments"]))
            if rec["tier"] != "none":
                posts[rec["pid"]] = rec
        print(f"[resample] {name}: {len(idx)} window posts", flush=True)

    if not posts:
        sys.exit("[resample] 0 posts — check --dumps-dir (try .../subreddits25)")

    # 2) scenario-classify all posts
    pids = list(posts)
    labels = classify([posts[p]["text"] for p in pids])
    for p, s in zip(pids, labels):
        posts[p]["scenario"] = s

    # 3) per (tier x scenario) quota fill: prefer deeper, then higher top score
    cells = defaultdict(list)  # (tier,scenario) -> [pid]
    for p in pids:
        cells[(posts[p]["tier"], posts[p]["scenario"])].append(p)

    def keyf(p):
        cs = posts[p]["comments"]
        return (len(cs), max((c["score"] for c in cs), default=0))

    chosen = []
    plan = []
    for tier in ["shallow", "mid", "deep"]:
        for s in scen_targets:
            pool = sorted(cells.get((tier, s), []), key=keyf, reverse=True)
            take = pool[:quota[tier]]
            chosen.extend(take)
            plan.append({"tier": tier, "scenario": s, "available": len(pool),
                         "quota": quota[tier], "taken": len(take)})

    # 4) emit comment rows (deep = keep all, else top max_per_post by score)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    tier_rows = Counter()
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS); w.writeheader()
        for p in chosen:
            rec = posts[p]
            cms = sorted(rec["comments"], key=lambda x: -x["score"])
            if rec["tier"] != "deep":
                cms = cms[:args.max_per_post]
            for cm in cms:
                w.writerow({
                    "annotation_id": f"rs_{n_rows:06d}", "post_id": p,
                    "comment_id": cm["comment_id"], "target_text": cm["body"],
                    "post_context": rec["text"], "subreddit": rec["sub"],
                    "comment_score": cm["score"], "comment_chars": len(cm["body"]),
                    "post_chars": len(rec["text"]), "tier": rec["tier"],
                    "scenario": rec["scenario"], "created_year": rec["year"]})
                n_rows += 1
                tier_rows[rec["tier"]] += 1

    manifest = {"window": f"{args.year_min}-{args.year_max}", "backend": args.backend,
                "model": args.model, "model_revision": args.model_revision,
                "embedding_batch_size": args.embedding_batch_size,
                "seed": args.seed,
                "selection_rule": (
                    "deterministic quota by tier x scenario; descending "
                    "qualifying-comment count then top-comment score; stable "
                    "source-stream order breaks exact ties"
                ),
                "quota": quota, "max_per_post_non_deep": args.max_per_post,
                "filters": {
                    "subreddits": args.subreddits,
                    "year_min": args.year_min,
                    "year_max": args.year_max,
                    "min_post_chars": args.min_post_chars,
                    "min_comment_chars": args.min_comment_chars,
                },
                "sources": {
                    "dumps_dir": str(args.dumps_dir),
                    "text_csv": str(args.text_csv),
                    "post_jsonl": str(args.post_jsonl),
                    "exclude_jsonl": str(exclude_path),
                },
                "n_posts_chosen": len(chosen), "n_comment_rows": n_rows,
                "rows_by_tier": dict(tier_rows),
                "posts_by_tier": dict(Counter(posts[p]["tier"] for p in chosen)),
                "posts_by_scenario": dict(Counter(posts[p]["scenario"] for p in chosen)),
                "cell_plan": plan}
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(f"\n[resample] chose {len(chosen)} posts / {n_rows} comment rows")
    print(f"  posts by tier: {manifest['posts_by_tier']}")
    print(f"  rows by tier:  {dict(tier_rows)}")
    print(f"saved -> {out_path}\nsaved -> {args.manifest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps-dir", type=Path, default=Path("../adhd_data_reddit/subreddits25"))
    ap.add_argument("--subreddits", default="ADHD")
    ap.add_argument("--year-min", type=int, default=2023)
    ap.add_argument("--year-max", type=int, default=2025)
    ap.add_argument("--shallow-q", type=int, default=40)
    ap.add_argument("--mid-q", type=int, default=25)
    ap.add_argument("--deep-q", type=int, default=20)
    ap.add_argument("--max-per-post", type=int, default=8,
                    help="cap for shallow/mid posts; deep posts keep ALL comments")
    ap.add_argument("--min-post-chars", type=int, default=300)
    ap.add_argument("--min-comment-chars", type=int, default=150)
    ap.add_argument("--text-csv", type=Path,
                    default=Path("out/annotation_input_merged_v2.csv"))
    ap.add_argument("--post-jsonl", type=Path,
                    default=Path("out/llm_post_problem_annotations_full.jsonl"))
    ap.add_argument("--exclude-jsonl", type=Path, default=None,
                    help=("post_id JSONL to exclude from the new draw. Defaults to "
                          "--post-jsonl for legacy behavior. Use this to classify "
                          "with a larger labelled pool while excluding only the "
                          "current knowledge-base posts."))
    ap.add_argument("--backend", default="bert", choices=["bert", "tfidf"])
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument(
        "--model-revision",
        default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    )
    ap.add_argument("--embedding-batch-size", type=int, default=64)
    ap.add_argument(
        "--seed", type=int, default=20260610,
        help=(
            "runtime seed for deterministic embedding inference; the quota/rank "
            "selection rule itself is deterministic"
        ),
    )
    ap.add_argument("--out", type=Path, default=Path("out/resample/resample_input_raw.csv"))
    ap.add_argument("--manifest", type=Path, default=Path("out/resample/resample_manifest.json"))
    a = ap.parse_args()
    run(a)


if __name__ == "__main__":
    main()
