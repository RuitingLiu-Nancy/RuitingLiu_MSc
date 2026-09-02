#!/usr/bin/env python3
"""Open lived-experience entity extraction for ADHD Reddit comments.

This is intentionally looser than llm_extract_entities_v2.py:

- It does NOT force every mention into the old 8-type closed taxonomy.
- It keeps the model's first-pass wording and a light `loose_kind`.
- It requires evidence spans so the extracted item can be traced to comment text
  or the paired post context.
- It emits raw JSONL suitable for later clustering/canonicalisation.

Recommended use:

  cd /path/to/project

  # Cheap pilot
  python -m data_preparation.entity_processing.open_entity_extraction \
    --input out/resample/entity_pilot_strategy_2000.csv \
    --out out/resample/open_entities_pilot_50_haiku.jsonl \
    --model claude:claude-haiku-4-5 \
    --limit 50

  # Larger run after checking the pilot
  python -m data_preparation.entity_processing.open_entity_extraction \
    --input out/resample/entity_pilot_strategy_2000.csv \
    --out out/resample/open_entities_strategy_2000_haiku.jsonl \
    --model claude:claude-haiku-4-5 \
    --entities-only
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

def system_prompt() -> str:
    """Load the system instruction from the external prompt store."""
    import configuration as config
    return config.prompt("openie_system")


def build_prompt(post_context: str, comment_text: str, entities_only: bool = False) -> str:
    """Render the external OpenIE request template without bundling its text."""
    import configuration as config
    template = config.prompt("openie_user")
    return template.format(
        post_context=post_context,
        comment_text=comment_text,
        entities_only=json.dumps(bool(entities_only)),
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if path.suffix == ".parquet":
        return pd.read_parquet(path).fillna("").to_dict("records")
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna("").to_dict("records")


def to_record(row: dict[str, Any]) -> dict[str, str]:
    comment_id = str(row.get("comment_id") or row.get("unit_id") or "").strip()
    post_id = str(row.get("post_id") or "").strip()
    text = str(row.get("target_text") or row.get("text") or row.get("body") or "").strip()
    return {
        "unit_id": comment_id or post_id,
        "unit_type": "comment" if comment_id else "post",
        "post_id": post_id,
        "comment_id": comment_id,
        "post_context": str(row.get("post_context") or row.get("problem_summary") or "")[:1800],
        "text": text[:3200],
        "scenario": str(row.get("scenario") or ""),
        "tier": str(row.get("tier") or ""),
        "support_functions": str(row.get("support_functions") or ""),
        "strategy_domains": str(row.get("strategy_domains") or ""),
        "ef_mechanisms": str(row.get("ef_mechanisms") or ""),
    }


def parse_json_object(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    return json.loads(text)


def build_json_retry_prompt(original_prompt: str, parse_error: Exception) -> str:
    import configuration as config
    template = config.prompt("openie_json_retry")
    return template.format(original_prompt=original_prompt, parse_error=str(parse_error))


_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def norm_text(value: object) -> str:
    value = _PUNCT.sub(" ", str(value).lower())
    return _WS.sub(" ", value).strip()


def clean_kind(value: object) -> str:
    value = str(value or "").strip().lower()
    return value if value in LOOSE_KINDS else "other_specific_concept"


def clean_relation(value: object) -> str:
    value = str(value or "").strip().lower().replace(" ", "_")
    return value if value in RELATION_HINTS else "co_occurs_with"


def normalize_response(obj: dict[str, Any], rec: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    text_norm = norm_text(rec["text"])
    post_norm = norm_text(rec["post_context"])

    entities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    canonicals: set[str] = set()
    raw_entities = obj.get("entities", [])
    if not isinstance(raw_entities, list):
        raw_entities = []

    for ent in raw_entities:
        if not isinstance(ent, dict):
            continue
        surface = str(ent.get("surface") or ent.get("text") or "").strip()
        canonical = str(ent.get("canonical") or surface).strip().lower()
        evidence = str(ent.get("evidence") or surface).strip()
        if not surface or not canonical:
            continue
        if not evidence or norm_text(evidence) not in text_norm:
            if norm_text(evidence) not in post_norm:
                warnings.append(f"evidence_not_found:{canonical}")
                evidence_source = str(ent.get("evidence_source") or "unknown").strip().lower()
            else:
                evidence_source = "post_context"
        else:
            evidence_source = "comment"
        key = (canonical, clean_kind(ent.get("loose_kind") or ent.get("kind")))
        if key in seen:
            continue
        seen.add(key)
        canonicals.add(canonical)
        entities.append({
            "surface": surface[:180],
            "canonical": canonical[:120],
            "loose_kind": key[1],
            "evidence_source": evidence_source,
            "evidence": evidence[:320],
            "why_useful": str(ent.get("why_useful") or "")[:260],
        })

    relations: list[dict[str, str]] = []
    seen_rel: set[tuple[str, str, str]] = set()
    raw_relations = obj.get("relations", [])
    if not isinstance(raw_relations, list):
        raw_relations = []

    # GENERATE stage (GraphRAG / Graphusion "extract-then-resolve"): keep ALL
    # relation instances. We DO NOT discard a relation just because an endpoint
    # was not extracted as an entity in THIS comment — that missing endpoint is
    # very often extracted elsewhere and will be unified during GLOBAL entity
    # resolution (canonicalisation). Discarding here causes the well-known
    # "fragmented graph" failure mode. Instead we tag each relation with whether
    # both endpoints were locally grounded, so the global-grounding step can
    # audit and (re)attach them to canonical nodes.
    for rel in raw_relations:
        if not isinstance(rel, dict):
            continue
        source = str(rel.get("source") or "").strip().lower()
        target = str(rel.get("target") or "").strip().lower()
        relation = clean_relation(rel.get("relation"))
        if not source or not target:
            continue
        grounded_local = (source in canonicals and target in canonicals)
        if not grounded_local:
            # informational only — relation is RETAINED, not dropped
            warnings.append(f"relation_endpoint_not_local:{source}->{target}")
        key = (source, relation, target)
        if key in seen_rel:
            continue
        seen_rel.add(key)
        relations.append({
            "source": source[:120],
            "relation": relation,
            "target": target[:120],
            "grounded_local": grounded_local,
            "evidence_source": str(rel.get("evidence_source") or "")[:40],
            "evidence": str(rel.get("evidence") or "")[:320],
        })

    try:
        confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.0))))
    except Exception:
        confidence = 0.0

    clean = {
        "unit_id": rec["unit_id"],
        "unit_type": rec["unit_type"],
        "post_id": rec["post_id"],
        "comment_id": rec["comment_id"],
        "scenario": rec["scenario"],
        "tier": rec["tier"],
        "support_functions": rec["support_functions"],
        "strategy_domains": rec["strategy_domains"],
        "ef_mechanisms": rec["ef_mechanisms"],
        "comment_summary": str(obj.get("comment_summary") or "")[:500],
        "entities": entities,
        "relations": relations,
        "confidence": confidence,
    }
    return clean, warnings


def load_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    good: dict[str, str] = {}
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        uid = str(row.get("unit_id") or "")
        if not uid:
            continue
        if "LLM/parse error" in str(row.get("_warnings", "")):
            continue
        good[uid] = line
    out_path.write_text("".join(line + "\n" for line in good.values()), encoding="utf-8")
    return set(good)


def is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(x in text for x in ["429", "rate limit", "rate_limit", "too many requests"])


def is_fatal_call_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(x in text for x in [
        "resourcenotfoundexception",
        "validationexception",
        "accessdenied",
        "not authorized",
        "unable to locate credentials",
        "no credential",
        "end of its life",
    ])


def call_chat_retry(call_chat, prompt: str, *, model_spec: str, max_tokens: int,
                    temperature: float, max_retries: int = 6) -> str:
    delay = 5.0
    for attempt in range(max_retries + 1):
        try:
            return call_chat(prompt, model_spec=model_spec, max_tokens=max_tokens,
                             temperature=temperature)
        except Exception as exc:
            if not is_rate_limit(exc) or attempt == max_retries:
                raise
            sleep_s = delay + random.uniform(0, delay * 0.25)
            print(f"[rate-limit] attempt {attempt + 1}, sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)
            delay = min(delay * 2, 120.0)
    raise RuntimeError("unreachable")


def run(input_path: Path, out_path: Path, model_spec: str, limit: int | None,
        max_tokens: int, entities_only: bool = False) -> None:
    from shared.llm_client import call_chat

    rows = read_rows(input_path)
    if limit:
        rows = rows[:limit]
    records = [to_record(row) for row in rows]
    records = [rec for rec in records if rec["unit_id"] and rec["text"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(out_path)
    todo = [rec for rec in records if rec["unit_id"] not in done]
    if done:
        print(f"[open-entities] resume: {len(done)} done, {len(todo)} remaining")

    with out_path.open("a", encoding="utf-8") as fh:
        for i, rec in enumerate(todo, 1):
            prompt = system_prompt() + "\n\n" + build_prompt(
                rec["post_context"], rec["text"], entities_only=entities_only)
            try:
                response = call_chat_retry(
                    call_chat,
                    prompt,
                    model_spec=model_spec,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                try:
                    obj = parse_json_object(response)
                except Exception as parse_exc:
                    retry_prompt = build_json_retry_prompt(prompt, parse_exc)
                    response = call_chat_retry(
                        call_chat,
                        retry_prompt,
                        model_spec=model_spec,
                        max_tokens=max_tokens,
                        temperature=0.0,
                    )
                    obj = parse_json_object(response)
                clean, warnings = normalize_response(obj, rec)
            except Exception as exc:
                if is_fatal_call_error(exc):
                    raise RuntimeError(
                        f"fatal model call error for {model_spec}; stopping before "
                        f"writing empty extraction rows: {exc}"
                    ) from exc
                clean, warnings = normalize_response({}, rec)
                warnings.append(f"LLM/parse error: {exc}")
            clean["_warnings"] = "|".join(warnings)
            fh.write(json.dumps(clean, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                print(
                    f"[open-entities] {len(done) + i}/{len(records)} "
                    f"(last {len(clean['entities'])}E/{len(clean['relations'])}R)"
                )

    print(f"[open-entities] wrote -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("out/open_entities.jsonl"))
    parser.add_argument("--model", default="claude:claude-haiku-4-5")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--entities-only", action="store_true",
                        help="Only extract open entities; skip relation extraction.")
    parser.add_argument("--print-prompt", action="store_true")
    args = parser.parse_args()

    if args.print_prompt:
        print("===== SYSTEM =====")
        print(system_prompt())
        print("\n===== USER =====")
        print(build_prompt(
            "I can never bring myself to study even when the topic is interesting.",
            "I use a timer and talk through the essay first. I dictate a messy "
            "draft, then clean it up later. Body doubling also helps me start.",
            entities_only=args.entities_only,
        ))
        return

    if not args.input:
        parser.error("--input is required unless --print-prompt is used")
    run(args.input, args.out, args.model, args.limit, args.max_tokens,
        entities_only=args.entities_only)


if __name__ == "__main__":
    main()
