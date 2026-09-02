#!/usr/bin/env python3
"""Batch-API version of open-entity extraction (OpenAI ONLY).

This script implements the OpenAI Batch transport. Other hosted providers can
reuse the same external prompt and parser through their own transport adapter.

Uses OpenAI's Batch API (50% cheaper, no synchronous RPM/TPM rate-limits) for
the large full-corpus extraction. It REUSES the exact prompt + parsing +
normalisation from llm_extract_open_entities.py, so output is identical to the
per-request version — just produced asynchronously and cheaply.

Three steps (run them in order):

  1) prepare : build the batch input JSONL (one /v1/chat/completions request per
               comment). No API call. Skips comments already in --out (resume).
        python extract_entities_batch.py prepare \
          --input out/resample/resample_comment_input.csv \
          --batch-input out/resample/batch_input.jsonl \
          --out out/resample/open_entities_full_4omini.jsonl \
          --model gpt-4o-mini --max-tokens 800

  2) submit  : upload the file + create the batch. Prints a batch_id.
        python extract_entities_batch.py submit \
          --batch-input out/resample/batch_input.jsonl

  3) fetch   : poll the batch; when complete, download results, parse with the
               SAME normalize_response, and APPEND to --out (resume-safe).
        python extract_entities_batch.py fetch \
          --batch-id batch_xxx \
          --input out/resample/resample_comment_input.csv \
          --out out/resample/open_entities_full_4omini.jsonl

Notes
-----
- custom_id = unit_id, so results map back to the right comment regardless of
  output ordering.
- A batch can hold up to 50,000 requests / 200 MB — your ~12k comments fit in one.
- Batches usually finish in minutes–hours (SLA: within 24h).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# reuse the EXACT extraction logic
from data_preparation.entity_processing import open_entity_extraction as ext


def _records(input_path: Path):
    rows = ext.read_rows(input_path)
    recs = [ext.to_record(r) for r in rows]
    return [r for r in recs if r["unit_id"] and r["text"]]


# ---------------- 1) prepare ----------------
def cmd_prepare(args):
    recs = _records(Path(args.input))
    done = ext.load_done_ids(Path(args.out)) if Path(args.out).exists() else set()
    todo = [r for r in recs if r["unit_id"] not in done]
    print(f"[batch:prepare] {len(recs)} total, {len(done)} already done, "
          f"{len(todo)} to request")

    out = Path(args.batch_input)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for rec in todo:
            prompt = ext.system_prompt() + "\n\n" + ext.build_prompt(
                rec["post_context"], rec["text"], entities_only=args.entities_only)
            req = {
                "custom_id": rec["unit_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                },
            }
            fh.write(json.dumps(req, ensure_ascii=False) + "\n")
            n += 1
    print(f"[batch:prepare] wrote {n} requests -> {out}")
    if n == 0:
        print("[batch:prepare] nothing to do (all comments already extracted).")


# ---------------- 2) submit ----------------
def _client():
    from openai import OpenAI
    return OpenAI()


def cmd_submit(args):
    client = _client()
    up = client.files.create(file=open(args.batch_input, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"job": "adhd-open-entities"},
    )
    print(f"[batch:submit] file_id={up.id}")
    print(f"[batch:submit] batch_id={batch.id}  status={batch.status}")
    print("  -> next: python extract_entities_batch.py fetch --batch-id "
          f"{batch.id} --input <input.csv> --out <out.jsonl>")


# ---------------- 3) fetch ----------------
def cmd_fetch(args):
    client = _client()
    # poll
    while True:
        batch = client.batches.retrieve(args.batch_id)
        rc = getattr(batch, "request_counts", None)
        done = getattr(rc, "completed", "?") if rc else "?"
        total = getattr(rc, "total", "?") if rc else "?"
        print(f"[batch:fetch] status={batch.status}  {done}/{total}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        if args.no_wait:
            print("[batch:fetch] --no-wait set; exiting (not complete yet).")
            return
        time.sleep(args.poll_seconds)

    if batch.status != "completed":
        print(f"[batch:fetch] batch ended as {batch.status}.")
        if batch.error_file_id:
            err = client.files.content(batch.error_file_id).text
            Path(args.out).with_suffix(".errors.jsonl").write_text(err, encoding="utf-8")
            print("  wrote error file.")
        if batch.status != "completed":
            return

    # download results
    content = client.files.content(batch.output_file_id).text
    results = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj.get("custom_id")
        body = (((obj.get("response") or {}).get("body")) or {})
        choices = body.get("choices") or []
        text = choices[0]["message"]["content"] if choices else ""
        results[cid] = text
    print(f"[batch:fetch] downloaded {len(results)} results")

    # parse with the SAME normalize_response, keyed by unit_id
    rec_by_id = {r["unit_id"]: r for r in _records(Path(args.input))}
    done = ext.load_done_ids(Path(args.out)) if Path(args.out).exists() else set()
    written = 0
    with Path(args.out).open("a", encoding="utf-8") as fh:
        for cid, text in results.items():
            if cid in done:
                continue
            rec = rec_by_id.get(cid)
            if not rec:
                continue
            try:
                clean, warnings = ext.normalize_response(ext.parse_json_object(text), rec)
            except Exception as exc:
                clean, warnings = ext.normalize_response({}, rec)
                warnings.append(f"LLM/parse error: {exc}")
            clean["_warnings"] = "|".join(warnings)
            fh.write(json.dumps(clean, ensure_ascii=False) + "\n")
            written += 1
    print(f"[batch:fetch] appended {written} parsed records -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--input", required=True)
    p.add_argument("--batch-input", required=True)
    p.add_argument("--out", required=True, help="final jsonl (for resume skipping)")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--entities-only", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("submit")
    p.add_argument("--batch-input", required=True)
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("fetch")
    p.add_argument("--batch-id", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--no-wait", action="store_true")
    p.set_defaults(func=cmd_fetch)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
