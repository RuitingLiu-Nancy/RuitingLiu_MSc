#!/usr/bin/env python3
"""Fail-closed preparation for coverage-complete drift-rescue judging.

``preflight`` is local-only.  It verifies the two report-91 residual manifests,
freezes a deduplicated ADMIN mapping, and emits provider-visible payloads that
contain only the canonical utility-v2 fields.

``judge`` is implemented only as the explicitly authorised continuation.  It
requires both the checked-in configuration gate and a command-line consent
flag.  The default configuration is false, so running this file without a
later explicit authorisation cannot call an external model.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import shutil
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.judgment_completeness import (
    DIMS_V2,
    complete_utility_v2_rows,
)
from evaluation.utility import (
    annotation_prompt as _prompt,
    utility_v2 as _utility_v2,
)
from utility_scoring.annotation import run_top3_residual_judging as canonical_judge
from utility_scoring.annotation.run_top3_residual_judging import (
    MAX_RETRIES,
    MAX_TOKENS,
    MODEL,
    TEMPERATURE,
    anchor_report,
    hash_json,
    judge_payload,
    load_corpus,
    load_queries,
    read_jsonl,
    reject_test_path as canonical_reject_test_path,
    sha256,
    utc_now,
    validated_existing,
    write_json,
    write_jsonl,
)


CONFIG_KEY = "coverage_complete_residual_judging"
PAYLOAD_FIELDS = ("query_text", "comment_text", "facets_json")
FORBIDDEN_PAYLOAD_TOKENS = (
    "query_id",
    "comment_id",
    "source",
    "rank",
    "score",
    "similarity",
    "utility",
    "drift",
    "action",
    "oracle",
    "graph",
    "bm25",
)
URL_ONLY = re.compile(
    r"^\s*(?:(?:https?://|www\.)\S+)(?:\s+(?:(?:https?://|www\.)\S+))*\s*$",
    re.IGNORECASE,
)
TEST_PATH = re.compile(r"(^|[/_.-])(?:frozen[_-]?)?test(?:200)?($|[/_.-])", re.I)


def exact_pairs(rows: list[dict], label: str) -> list[tuple[str, str]]:
    pairs = [
        (str(row.get("query_id") or ""), str(row.get("comment_id") or ""))
        for row in rows
    ]
    if any(not query_id or not comment_id for query_id, comment_id in pairs):
        raise ValueError(f"{label}: missing pair identity")
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"{label}: duplicate pair identity")
    return pairs


def read_json_object(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_exclusive(path: Path, value) -> None:
    """Write a versioned audit artefact once; never replace an earlier record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def reject_drift_judging_test_path(path: Path) -> None:
    canonical_reject_test_path(path)
    if TEST_PATH.search(str(path.resolve())):
        raise ValueError(f"frozen-test-looking path rejected: {path}")


def load_preflight_config(
    root: Path,
    path: Path,
    config_key: str = CONFIG_KEY,
) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config_key not in raw:
        raise KeyError(f"missing config block: {config_key}")
    cfg = dict(raw[config_key])
    cfg["_root"] = root
    return cfg


def resolve_cfg_path(cfg: dict, key: str) -> Path:
    path = Path(cfg[key])
    return path if path.is_absolute() else cfg["_root"] / path


def provider_payload(query_text: str, comment_text: str) -> dict:
    payload = {
        "query_text": query_text,
        "comment_text": comment_text,
        "facets_json": {},
    }
    assert_payload_blind(payload)
    return payload


def assert_payload_blind(payload: dict) -> None:
    if set(payload) != set(PAYLOAD_FIELDS) or len(payload) != len(PAYLOAD_FIELDS):
        raise ValueError(f"provider payload schema changed: {sorted(payload)}")
    if payload["facets_json"] != {}:
        raise ValueError("canonical utility-v2 requires facets_json={}")
    if not payload["query_text"].strip() or not payload["comment_text"].strip():
        raise ValueError("provider payload contains empty text")
    if URL_ONLY.fullmatch(payload["comment_text"]):
        raise ValueError("provider payload contains unresolved URL-only comment")
    forbidden = set(payload) & set(FORBIDDEN_PAYLOAD_TOKENS)
    if forbidden:
        raise ValueError(f"forbidden provider payload fields: {sorted(forbidden)}")


def protocol_lock(prompt: Path, registry: list[dict]) -> dict:
    complete, _ = complete_utility_v2_rows(registry)
    schema = {
        "type": "object",
        "required": [*DIMS_V2, "rationale"],
        "properties": {
            **{
                dim: {"type": "integer", "minimum": 1, "maximum": 7}
                for dim in DIMS_V2
            },
            "rationale": {"type": "string"},
        },
        "additionalProperties": False,
    }
    utility_examples = {
        "all_ones": _utility_v2({dim: 1 for dim in DIMS_V2}),
        "all_sevens": _utility_v2({dim: 7 for dim in DIMS_V2}),
        "safety_gate": _utility_v2({
            "relevance": 7,
            "usefulness": 7,
            "novelty": 7,
            "actionability": 7,
            "resonance": 7,
            "safety": 2,
        }),
    }
    return {
        "verdict": "LOCKED_TO_EXISTING_IMPLEMENTATION",
        "implementation_identity": {
            "prompt_builder": (
                "study_platform.backend.llm_annotate_evidence._prompt"
            ),
            "request_executor": (
                "tools.run_top3_residual_judging.judge_payload"
            ),
            "utility_function": "tools.run_section17_pipeline._utility_v2",
            "anchor_gate": "tools.run_top3_residual_judging.anchor_report",
        },
        "system_prompt": None,
        "user_prompt": {
            "path": str(prompt),
            "sha256": sha256(prompt),
        },
        "provider_visible_input_fields": list(PAYLOAD_FIELDS),
        "facets_json_fixed_value": {},
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "max_retries": MAX_RETRIES,
        "retry_backoff_seconds": {"initial": 2.0, "maximum": 30.0},
        "parser": "json -> first JSON object -> dimension regex fallback",
        "score_validation": "round/clamp each dimension to integer 1..7",
        "json_schema": schema,
        "json_schema_sha256": hash_json(schema),
        "utility": {
            "formula": (
                "0.25R + 0.30H + 0.15A + 0.10N + 0.10E + 0.10S"
            ),
            "safety_gate": "if S <= 2, utility=min(linear_utility, 2.0)",
            "implementation_examples": utility_examples,
        },
        "historical_registry": {
            "complete_pairs": len(complete),
            "models": sorted({str(row.get("judge_model")) for row in complete}),
            "judge_ids": sorted({str(row.get("judge_id")) for row in complete}),
        },
        "implementation_source_sha256": {
            "prompt_builder": hashlib.sha256(
                inspect.getsource(_prompt).encode("utf-8")
            ).hexdigest(),
            "request_executor": hashlib.sha256(
                inspect.getsource(judge_payload).encode("utf-8")
            ).hexdigest(),
            "utility_function": hashlib.sha256(
                inspect.getsource(_utility_v2).encode("utf-8")
            ).hexdigest(),
            "anchor_gate": hashlib.sha256(
                inspect.getsource(anchor_report).encode("utf-8")
            ).hexdigest(),
        },
    }


def token_estimate(payload_rows: list[dict]) -> dict:
    estimated_input = sum(
        (
            len(row["query_text"])
            + len(row["comment_text"])
            + 1600
        ) // 4
        for row in payload_rows
    )
    estimated_output = 120 * len(payload_rows)
    return {
        "logical_items": len(payload_rows),
        "provider_items_per_request": 1,
        "first_pass_provider_requests": len(payload_rows),
        "maximum_provider_attempts_including_retries": (
            len(payload_rows) * (MAX_RETRIES + 1)
        ),
        "estimated_input_tokens": estimated_input,
        "estimated_output_tokens": estimated_output,
        "token_estimator": (
            "(query_chars + comment_chars + 1600) // 4 input; "
            "120 output tokens/item, reused from run_section17_pipeline"
        ),
        "usd_estimate": None,
        "usd_estimate_status": (
            "UNAVAILABLE_NO_FROZEN_LLAMA_3_3_PRICE_IN_PROJECT_SOURCE_OF_TRUTH"
        ),
    }


def run_preflight(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    cfg = load_preflight_config(root, args.config.resolve())
    paths = {
        key: resolve_cfg_path(cfg, key)
        for key in (
            "report91_manifest",
            "report91_v1_superseded",
            "strict_graph_report",
            "bm25_residual",
            "graph_residual",
            "utility_registry",
            "queries",
            "corpus",
            "anchors",
            "prompt",
        )
    }
    out_dir = resolve_cfg_path(cfg, "preflight_output_dir")
    for path in [*paths.values(), out_dir]:
        reject_drift_judging_test_path(path)
    required_outputs = (
        "residual_union_manifest.jsonl",
        "residual_overlap_report.json",
        "already_judged_exclusions.jsonl",
        "invalid_or_unresolved_pairs.jsonl",
        "judging_cost_and_call_estimate.json",
        "preflight_manifest.json",
        "preflight_report.md",
        "residual_judging_payload.jsonl",
        "anchor_payload.jsonl",
        "anchor_payload_admin.jsonl",
        "judge_protocol_lock.json",
        "payload_blindness_audit.json",
        "execution_plan.md",
        "exact_execution_command.txt",
    )
    if out_dir.exists() and any((out_dir / name).exists() for name in required_outputs):
        raise FileExistsError(
            f"refusing to overwrite frozen preflight artefacts in {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    violations: list[str] = []
    report91 = read_json_object(paths["report91_manifest"])
    if report91.get("version") != cfg["authoritative_report91_version"]:
        violations.append("report91 authoritative version mismatch")
    if report91.get("test_read") is not False:
        violations.append("report91 does not attest test_read=false")
    superseded = paths["report91_v1_superseded"].read_text(encoding="utf-8")
    if "SUPERSEDED" not in superseded:
        violations.append("report91 v1 is not explicitly superseded")

    expected_hashes = report91.get("output_hashes", {})
    for label, path in (
        ("residual_bm25_judgment_manifest.jsonl", paths["bm25_residual"]),
        (
            "residual_graph_practical_judgment_manifest.jsonl",
            paths["graph_residual"],
        ),
    ):
        if expected_hashes.get(label) != sha256(path):
            violations.append(f"report91 output hash mismatch: {label}")
    report_inputs = report91.get("input_hashes", {})
    for path in (paths["utility_registry"], paths["strict_graph_report"]):
        rel = str(path.relative_to(root))
        if report_inputs.get(rel) != sha256(path):
            violations.append(f"report91 input hash mismatch: {rel}")

    bm25 = read_jsonl(paths["bm25_residual"])
    graph = read_jsonl(paths["graph_residual"])
    expected_counts = cfg["expected_counts"]
    if len(bm25) != int(expected_counts["bm25_raw"]):
        violations.append(f"BM25 residual count changed: {len(bm25)}")
    if len(graph) != int(expected_counts["graph_raw"]):
        violations.append(f"Graph residual count changed: {len(graph)}")
    try:
        bm25_pairs = exact_pairs(bm25, "BM25 residual")
        graph_pairs = exact_pairs(graph, "Graph residual")
    except ValueError as exc:
        violations.append(str(exc))
        bm25_pairs, graph_pairs = [], []

    registry_rows = read_jsonl(paths["utility_registry"])
    complete_rows, registry = complete_utility_v2_rows(registry_rows)
    if len(complete_rows) != int(expected_counts["registry_complete"]):
        violations.append(
            f"authoritative registry count changed: {len(complete_rows)}"
        )
    if {str(row.get("judge_model")) for row in complete_rows} != {MODEL}:
        violations.append("historical registry model differs from canonical model")
    if {str(row.get("judge_id")) for row in complete_rows} != {"utility-v2"}:
        violations.append("historical registry protocol differs from utility-v2")

    queries = load_queries(paths["queries"])
    corpus = load_corpus(paths["corpus"])
    by_pair: dict[tuple[str, str], dict] = {}
    text_conflicts: list[dict] = []
    raw_memberships: dict[tuple[str, str], list[str]] = {}
    raw_rows = [("bm25", row) for row in bm25] + [
        ("graph_practical", row) for row in graph
    ]
    for source, row in raw_rows:
        key = (str(row.get("query_id") or ""), str(row.get("comment_id") or ""))
        raw_memberships.setdefault(key, []).append(source)
        if key in by_pair and (
            by_pair[key].get("query_text") != row.get("query_text")
            or by_pair[key].get("comment_text") != row.get("comment_text")
        ):
            text_conflicts.append({
                "query_id": key[0],
                "comment_id": key[1],
                "reason": "cross_manifest_text_conflict",
            })
        else:
            by_pair[key] = row

    overlap = sorted(set(bm25_pairs) & set(graph_pairs))
    raw_union = sorted(set(bm25_pairs) | set(graph_pairs))
    already = sorted(set(raw_union) & set(registry))
    pending = sorted(set(raw_union) - set(registry))

    invalid: list[dict] = list(text_conflicts)
    union_admin: list[dict] = []
    provider_rows: list[dict] = []
    for index, (qid, cid) in enumerate(pending):
        reasons = []
        row = by_pair.get((qid, cid), {})
        if qid not in queries:
            reasons.append("query_id_missing_from_frozen_dev100")
        if cid not in corpus:
            reasons.append("comment_id_missing_from_frozen_corpus")
        query_text = queries.get(qid, "")
        comment_text = corpus.get(cid, "")
        if row.get("query_text") != query_text:
            reasons.append("query_text_mismatch_against_frozen_source")
        if row.get("comment_text") != comment_text:
            reasons.append("comment_text_mismatch_against_frozen_source")
        if not query_text.strip():
            reasons.append("empty_query_text")
        if not comment_text.strip():
            reasons.append("empty_comment_text")
        if URL_ONLY.fullmatch(comment_text):
            reasons.append("unresolved_url_only_comment")
        if reasons:
            invalid.append({
                "query_id": qid,
                "comment_id": cid,
                "reasons": reasons,
            })
            continue
        payload = provider_payload(query_text, comment_text)
        provider_rows.append(payload)
        union_admin.append({
            "payload_index": index,
            "query_id": qid,
            "comment_id": cid,
            "query_text": query_text,
            "comment_text": comment_text,
            "residual_sources": sorted(set(raw_memberships[(qid, cid)])),
            "source_pools": sorted(set(row.get("source_pools") or [])),
            "requested_rubric": row.get("requested_rubric"),
            "external_call_authorised": False,
            "test_split": False,
            "provider_payload_sha256": hash_json(payload),
        })

    anchors = read_jsonl(paths["anchors"])
    try:
        anchor_pairs = exact_pairs(anchors, "calibration anchors")
    except ValueError as exc:
        violations.append(str(exc))
        anchor_pairs = []
    if len(anchors) != int(expected_counts["anchors"]):
        violations.append(f"calibration anchor count changed: {len(anchors)}")
    if not set(anchor_pairs) <= set(registry):
        violations.append("one or more anchors lack historical utility-v2 judgments")
    if set(anchor_pairs) & set(pending):
        violations.append("calibration anchors overlap residual pairs")
    anchor_payloads: list[dict] = []
    anchor_admin: list[dict] = []
    for index, (qid, cid) in enumerate(anchor_pairs):
        if qid not in queries or cid not in corpus:
            violations.append(f"anchor text source unresolved: {(qid, cid)}")
            continue
        payload = provider_payload(queries[qid], corpus[cid])
        anchor_payloads.append(payload)
        anchor_admin.append({
            "payload_index": index,
            "query_id": qid,
            "comment_id": cid,
            "historical_utility": float(registry[(qid, cid)]["utility"]),
            "historical_scores": {
                dim: int(registry[(qid, cid)][f"label_{dim}"])
                for dim in DIMS_V2
            },
            "provider_payload_sha256": hash_json(payload),
            "selection_used_old_scores": bool(
                anchors[index].get("selection_used_old_scores")
            ),
        })

    if len(raw_union) != int(expected_counts["raw_union"]):
        violations.append(f"unexpected raw union count: {len(raw_union)}")
    if len(pending) != int(expected_counts["final_unique_pending"]):
        violations.append(f"unexpected final pending count: {len(pending)}")
    if invalid:
        violations.append(f"invalid or unresolved pair count: {len(invalid)}")
    if len(union_admin) != len(pending):
        violations.append("valid union payload count differs from pending pairs")
    if any(row.get("test_split") is not False for _, row in raw_rows):
        violations.append("a residual row does not attest test_split=false")

    protocol = protocol_lock(paths["prompt"], registry_rows)
    configured_judge = cfg.get("judge") or {}
    expected_judge = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "max_retries": MAX_RETRIES,
        "provider_items_per_request": 1,
    }
    if configured_judge != expected_judge:
        violations.append(
            f"configured judge differs from canonical protocol: "
            f"{configured_judge!r} != {expected_judge!r}"
        )
    protocol["config_lock"] = {
        "verdict": (
            "MATCHES_CANONICAL" if configured_judge == expected_judge
            else "PROTOCOL_MISMATCH"
        ),
        "configured": configured_judge,
        "canonical": expected_judge,
    }
    if protocol["historical_registry"]["models"] != [MODEL]:
        violations.append("protocol lock historical model mismatch")
    residual_cost = token_estimate(provider_rows)
    anchor_cost = token_estimate(anchor_payloads)
    total_cost = token_estimate(provider_rows + anchor_payloads)
    cost = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "max_retries": MAX_RETRIES,
        "residual": residual_cost,
        "calibration_anchors": anchor_cost,
        "total": total_cost,
        "pricing_source": "configuration/params.yaml::llm_pricing_usd_per_million_tokens",
        "pricing_lookup_model_key": "meta_llama_3_3_70b_instruct",
        "pricing_key_present": False,
        "instruction": (
            "Verify current official Bedrock price and region before execution; "
            "do not substitute another model."
        ),
    }
    overlap_report = {
        "bm25_raw_rows": len(bm25),
        "bm25_unique_pairs": len(set(bm25_pairs)),
        "graph_raw_rows": len(graph),
        "graph_unique_pairs": len(set(graph_pairs)),
        "cross_manifest_overlap_pairs": len(overlap),
        "cross_manifest_overlap": [
            {"query_id": qid, "comment_id": cid} for qid, cid in overlap
        ],
        "raw_union_pairs": len(raw_union),
        "already_complete_in_authoritative_registry": len(already),
        "final_unique_pending_pairs": len(pending),
        "pair_source_membership": dict(Counter(
            "+".join(sorted(set(raw_memberships[pair]))) for pair in raw_union
        )),
    }
    exclusions = [
        {
            "query_id": qid,
            "comment_id": cid,
            "exclusion_reason": "already_complete_in_authoritative_registry",
        }
        for qid, cid in already
    ]
    blindness = {
        "verdict": "PASS" if not invalid else "FAIL",
        "payload_allowlist": list(PAYLOAD_FIELDS),
        "payload_items_audited": len(provider_rows) + len(anchor_payloads),
        "residual_payload_items": len(provider_rows),
        "anchor_payload_items": len(anchor_payloads),
        "facets_json_fixed_value": {},
        "forbidden_metadata_not_sent": list(FORBIDDEN_PAYLOAD_TOKENS),
        "admin_mapping_is_local_only": True,
        "route_rank_score_present_in_provider_payload": False,
        "test_split_used": False,
    }

    write_jsonl(out_dir / "residual_union_manifest.jsonl", union_admin)
    write_json(out_dir / "residual_overlap_report.json", overlap_report)
    write_jsonl(out_dir / "already_judged_exclusions.jsonl", exclusions)
    write_jsonl(out_dir / "invalid_or_unresolved_pairs.jsonl", invalid)
    write_json(out_dir / "judging_cost_and_call_estimate.json", cost)
    write_jsonl(out_dir / "residual_judging_payload.jsonl", provider_rows)
    write_jsonl(out_dir / "anchor_payload.jsonl", anchor_payloads)
    write_jsonl(out_dir / "anchor_payload_admin.jsonl", anchor_admin)
    write_json(out_dir / "judge_protocol_lock.json", protocol)
    write_json(out_dir / "payload_blindness_audit.json", blindness)

    ready = not violations
    execution_command = (
        f"PYTHONPATH={root} python {root / 'utility_scoring/annotation/run_coverage_complete_residual_judging.py'} "
        f"judge --root {root} --config {args.config.resolve()} "
        "--allow-external-judging --workers 4"
    )
    (out_dir / "exact_execution_command.txt").write_text(
        execution_command + "\n", encoding="utf-8"
    )
    execution_plan = f"""# Coverage-complete residual judging execution plan

Status: `{"READY_FOR_EXPLICIT_EXTERNAL_AUTHORIZATION" if ready else "BLOCKED_BY_PREFLIGHT"}`

No external request was made during this preflight.  The checked-in
`allow_external_judging` gate is `{str(bool(cfg.get("allow_external_judging"))).lower()}`.

1. Obtain a new explicit user authorisation for sending the frozen development
   payload and 50 anchors to `{MODEL}`.
2. Change only `coverage_complete_residual_judging.allow_external_judging` to
   `true`; do not change model, prompt, payload, rubric, weights or residual set.
3. Re-run `preflight` into a new versioned directory if any frozen input hash
   changes.  Do not overwrite this directory.
4. Verify current model accessibility and official Bedrock pricing.  If this
   exact model is unavailable, stop; do not substitute another model.
5. Run the exact command in `exact_execution_command.txt`.  Both the config
   gate and `--allow-external-judging` are required.
6. Require 50 valid anchor rejudgments.  Reuse the historical
   `anchor_report` stability rules.  Stop before merge/reanalysis if the
   verdict is `MATERIAL_BATCH_DRIFT` or `INSUFFICIENT_VALID_ANCHORS`.
7. Only after a passing anchor gate, create new versioned complete/v3 outputs.
   Never overwrite report91 or the 14,070-pair registry.

Planned first-pass provider requests: {total_cost["first_pass_provider_requests"]}
({len(provider_rows)} residual + {len(anchor_payloads)} anchors; one item/request).
USD cost is intentionally unresolved because the project price registry has no
frozen Meta Llama 3.3 entry.
"""
    (out_dir / "execution_plan.md").write_text(execution_plan, encoding="utf-8")

    input_hashes = {
        key: {"path": str(path), "sha256": sha256(path)}
        for key, path in paths.items()
    }
    preliminary_outputs = [
        name for name in required_outputs
        if name not in {"preflight_manifest.json", "preflight_report.md"}
    ]
    manifest = {
        "schema": "coverage-complete-residual-judging-preflight-v1",
        "version": cfg["version"],
        "created_utc": utc_now(),
        "phase_completed": "PHASE_0_TO_2_ONLY",
        "status": (
            "READY_FOR_EXPLICIT_EXTERNAL_AUTHORIZATION"
            if ready else "BLOCKED_BY_PREFLIGHT"
        ),
        "allow_external_judging": bool(cfg.get("allow_external_judging")),
        "external_requests_made": 0,
        "judge_executed": False,
        "merge_executed": False,
        "v3_reanalysis_executed": False,
        "thesis_edited": False,
        "authoritative_report91_version": report91.get("version"),
        "report91_v1_formal_input": False,
        "strict_graph_provenance_verified": not any(
            "strict_native_graph" in violation for violation in violations
        ),
        "test_read": False,
        "counts": {
            "bm25_raw": len(bm25),
            "graph_raw": len(graph),
            "cross_manifest_overlap": len(overlap),
            "raw_union": len(raw_union),
            "already_judged_exclusions": len(already),
            "invalid_or_unresolved": len(invalid),
            "final_unique_pending": len(provider_rows),
            "anchors": len(anchor_payloads),
            "planned_total_judge_items": len(provider_rows) + len(anchor_payloads),
        },
        "violations": violations,
        "input_hashes": input_hashes,
        "output_hashes": {
            name: sha256(out_dir / name) for name in preliminary_outputs
        },
    }
    write_json(out_dir / "preflight_manifest.json", manifest)
    report = f"""# Coverage-Complete Residual Judging Preflight

> 状态：`{manifest["status"]}`  
> 范围：Phase 0--2；development100；外部请求 0；frozen test 未读  
> 权威上游：`{report91.get("version")}`

## 1. 本轮裁决

本地预检{"通过" if ready else "未通过"}。由于
`ALLOW_EXTERNAL_JUDGING=false`，本轮在 payload、anchors、成本/调用量估算和精确执行命令
处停止；没有运行 judge、没有合并 registry、没有生成 report91 v3，也没有修改论文。

## 2. Residual inventory

| Item | Count |
|---|---:|
| BM25 residual raw/unique | {len(bm25)} / {len(set(bm25_pairs))} |
| Graph-practical residual raw/unique | {len(graph)} / {len(set(graph_pairs))} |
| Cross-manifest overlap | {len(overlap)} |
| Raw union | {len(raw_union)} |
| Already judged exclusions | {len(already)} |
| Invalid/unresolved | {len(invalid)} |
| Final unique residual payload | {len(provider_rows)} |
| Frozen calibration anchors | {len(anchor_payloads)} |
| Planned first-pass calls | {total_cost["first_pass_provider_requests"]} |

全部 residual 文本均与冻结 dev100 query 文件和 19,013-comment corpus 逐字核对。
Provider-visible JSONL 只有 `query_text`, `comment_text`, `facets_json={{}}`，不含
ID、来源、rank、score、similarity、utility、drift、action、oracle 或 Graph/BM25 标签。

## 3. 协议锁

- Model: `{MODEL}`
- Temperature: `{TEMPERATURE}`
- Max output tokens: `{MAX_TOKENS}`
- Retries: `{MAX_RETRIES}`（2s 指数退避，最高 30s）
- Scoring contract: supplied separately through the controlled runtime configuration
- Anchor gate: 复用历史 `anchor_report`，本轮尚未运行

## 4. 成本边界

估计输入 token 为 {total_cost["estimated_input_tokens"]:,}，估计输出 token 为
{total_cost["estimated_output_tokens"]:,}。项目价格单一真相源中没有冻结的 Meta
Llama 3.3 价格，因此没有编造 USD 数字；正式执行前必须核对官方区域价格。

## 5. 未执行的后续阶段

Phase 3--9 全部 deferred：外部判断、anchor stability、non-destructive registry merge、
report91 v3 完整重跑、claim reconciliation、论文 revision 与最终一致性更新。
"""
    (out_dir / "preflight_report.md").write_text(report, encoding="utf-8")
    if violations:
        raise SystemExit(f"PREFLIGHT FAILED: {violations}")
    print(json.dumps({
        "status": manifest["status"],
        "residual_pairs": len(provider_rows),
        "anchors": len(anchor_payloads),
        "planned_calls": total_cost["first_pass_provider_requests"],
        "external_requests_made": 0,
        "output_dir": str(out_dir),
    }, ensure_ascii=False, indent=2))


def run_authorized_judge(args: argparse.Namespace) -> None:
    """Explicitly authorised continuation; never reached with default config."""
    root = args.root.resolve()
    cfg = load_preflight_config(
        root,
        args.config.resolve(),
        args.config_key,
    )
    if not bool(cfg.get("allow_external_judging")):
        raise SystemExit(
            "EXTERNAL JUDGING BLOCKED: config allow_external_judging=false"
        )
    if not args.allow_external_judging:
        raise SystemExit(
            "EXTERNAL JUDGING BLOCKED: --allow-external-judging is required"
        )
    preflight_dir = resolve_cfg_path(cfg, "preflight_output_dir")
    complete_dir = resolve_cfg_path(cfg, "complete_output_dir")
    reject_drift_judging_test_path(preflight_dir)
    reject_drift_judging_test_path(complete_dir)
    manifest = read_json_object(preflight_dir / "preflight_manifest.json")
    if manifest.get("status") != "READY_FOR_EXPLICIT_EXTERNAL_AUTHORIZATION":
        raise SystemExit("EXTERNAL JUDGING BLOCKED: preflight did not pass")
    if manifest.get("external_requests_made") != 0:
        raise SystemExit("EXTERNAL JUDGING BLOCKED: unexpected preflight call state")
    for name, expected in manifest["output_hashes"].items():
        path = preflight_dir / name
        if not path.exists() or sha256(path) != expected:
            raise SystemExit(f"EXTERNAL JUDGING BLOCKED: changed preflight file {name}")

    union = read_jsonl(preflight_dir / "residual_union_manifest.jsonl")
    anchors = read_jsonl(preflight_dir / "anchor_payload_admin.jsonl")
    residual_payloads = read_jsonl(
        preflight_dir / "residual_judging_payload.jsonl"
    )
    anchor_payloads = read_jsonl(preflight_dir / "anchor_payload.jsonl")
    if len(union) != len(residual_payloads) or len(anchors) != len(anchor_payloads):
        raise SystemExit("EXTERNAL JUDGING BLOCKED: ADMIN/payload length mismatch")
    for row, payload in zip(union, residual_payloads):
        assert_payload_blind(payload)
        if hash_json(payload) != row["provider_payload_sha256"]:
            raise SystemExit("EXTERNAL JUDGING BLOCKED: residual payload hash mismatch")
    for row, payload in zip(anchors, anchor_payloads):
        assert_payload_blind(payload)
        if hash_json(payload) != row["provider_payload_sha256"]:
            raise SystemExit("EXTERNAL JUDGING BLOCKED: anchor payload hash mismatch")

    complete_dir.mkdir(parents=True, exist_ok=True)
    request_dir = complete_dir / "raw_requests"
    response_dir = complete_dir / "raw_responses"
    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = {
        "residual": complete_dir / "residual_judgments_raw.jsonl",
        "anchor": complete_dir / "anchor_rejudgments_raw.jsonl",
    }
    valid_paths = {
        "residual": complete_dir / "residual_judgments_validated.jsonl",
        "anchor": complete_dir / "anchor_rejudgments_validated.jsonl",
    }
    failure_path = complete_dir / "judgment_failures.jsonl"
    done = {
        kind: validated_existing(path) for kind, path in valid_paths.items()
    }
    prompt_sha = sha256(resolve_cfg_path(cfg, "prompt"))
    batch_id = str(cfg["batch_id"])
    audit_local = threading.local()
    original_call_chat = canonical_judge.call_chat

    def audited_call_chat(
        prompt: str,
        model_spec: str,
        max_tokens: int = 650,
        temperature: float = 0.35,
    ) -> str:
        """Observe canonical request attempts without changing judge_payload."""
        context = getattr(audit_local, "context", None)
        if context is None:
            raise RuntimeError("external call attempted without audit context")
        attempt = {
            "attempt_index": len(context["attempts"]),
            "started_utc": utc_now(),
            "model_spec": model_spec,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "rendered_prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
        }
        try:
            response = original_call_chat(
                prompt,
                model_spec=model_spec,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            attempt.update({
                "completed_utc": utc_now(),
                "status": "success",
                "raw_response": response,
                "raw_response_sha256": hashlib.sha256(
                    response.encode("utf-8")
                ).hexdigest(),
            })
            context["attempts"].append(attempt)
            return response
        except Exception as exc:
            attempt.update({
                "completed_utc": utc_now(),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            context["attempts"].append(attempt)
            raise

    # ``judge_payload`` resolves this module global at call time.  Replacing it
    # only with an audit-preserving wrapper leaves the frozen implementation,
    # prompt, parser, retry loop, model and inference parameters unchanged.
    canonical_judge.call_chat = audited_call_chat

    residual_items: list[tuple[str, dict, dict]] = []
    for admin, payload in zip(union, residual_payloads):
        key = (str(admin["query_id"]), str(admin["comment_id"]))
        if key not in done["residual"]:
            residual_items.append(("residual", admin, payload))
    anchor_items: list[tuple[str, dict, dict]] = []
    for admin, payload in zip(anchors, anchor_payloads):
        key = (str(admin["query_id"]), str(admin["comment_id"]))
        if key not in done["anchor"]:
            anchor_items.append(("anchor", admin, payload))

    handles = {
        path: path.open("a", encoding="utf-8")
        for path in [*raw_paths.values(), *valid_paths.values(), failure_path]
    }
    completed = failed = 0

    audit_paths: dict[tuple[str, str], dict] = {}

    def audited_judge_item(
        kind: str,
        admin: dict,
        payload: dict,
    ) -> tuple[dict, dict, dict]:
        payload_index = int(admin["payload_index"])
        payload_hash = str(admin["provider_payload_sha256"])
        stem = f"{kind}_{payload_index:04d}_{payload_hash[:12]}"
        request_path = request_dir / f"{stem}.json"
        response_path = response_dir / f"{stem}.json"
        rendered_prompt = _prompt(
            payload["query_text"], payload["comment_text"], {}, "v2"
        )
        request_body = {
            "modelId": MODEL.split(":", 1)[1],
            "messages": [{
                "role": "user",
                "content": [{"text": rendered_prompt}],
            }],
            "inferenceConfig": {
                "maxTokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
            },
        }
        write_json_exclusive(request_path, {
            "provider": "Amazon Bedrock Converse",
            "request_body": request_body,
            "request_body_sha256": hash_json(request_body),
            "provider_payload_sha256": payload_hash,
            "provider_payload_field_names": list(payload),
            "created_utc": utc_now(),
        })
        context = {"attempts": []}
        audit_local.context = context
        try:
            raw, valid = judge_payload(
                payload,
                "calibration_anchor" if kind == "anchor" else "residual",
                batch_id,
                prompt_sha,
                {
                    "query_id": str(admin["query_id"]),
                    "comment_id": str(admin["comment_id"]),
                },
            )
            response_audit = {
                "status": "valid",
                "validation_status": valid["validation_status"],
                "attempt_count": len(context["attempts"]),
                "retry_count": int(valid["retry_count"]),
                "attempts": context["attempts"],
                "validated_scores": valid["validated_scores"],
                "utility": valid["utility"],
                "completed_utc": utc_now(),
            }
            write_json_exclusive(response_path, response_audit)
            audit_paths[(
                str(admin["query_id"]), str(admin["comment_id"])
            )] = {
                "raw_request_path": str(request_path.relative_to(root)),
                "raw_response_path": str(response_path.relative_to(root)),
                "provider_payload_sha256": payload_hash,
                "request_body_sha256": hash_json(request_body),
                "completed_utc": response_audit["completed_utc"],
            }
            return raw, valid, response_audit
        except Exception:
            write_json_exclusive(response_path, {
                "status": "failed",
                "attempt_count": len(context["attempts"]),
                "attempts": context["attempts"],
                "completed_utc": utc_now(),
            })
            raise
        finally:
            audit_local.context = None

    def run_items(items: list[tuple[str, dict, dict]]) -> tuple[int, int]:
        local_completed = local_failed = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    audited_judge_item,
                    kind,
                    admin,
                    payload,
                ): (kind, admin)
                for kind, admin, payload in items
            }
            for future in as_completed(futures):
                kind, admin = futures[future]
                try:
                    raw, valid, _ = future.result()
                    handles[raw_paths[kind]].write(
                        json.dumps(raw, ensure_ascii=False) + "\n"
                    )
                    handles[valid_paths[kind]].write(
                        json.dumps(valid, ensure_ascii=False) + "\n"
                    )
                    handles[raw_paths[kind]].flush()
                    handles[valid_paths[kind]].flush()
                    local_completed += 1
                except Exception as exc:  # pragma: no cover - external continuation
                    handles[failure_path].write(json.dumps({
                        "query_id": str(admin["query_id"]),
                        "comment_id": str(admin["comment_id"]),
                        "item_type": kind,
                        "error": f"{type(exc).__name__}: {exc}",
                        "created_utc": utc_now(),
                    }, ensure_ascii=False) + "\n")
                    handles[failure_path].flush()
                    local_failed += 1
        return local_completed, local_failed

    try:
        # The user-fixed order is strict: complete and audit all 50 anchors
        # before any residual development text is sent.  ``anchors`` and
        # ``residual`` are separate resumable invocations so that the stability
        # verdict is persisted and inspectable before the residual stage starts.
        if args.stage in {"anchors", "all"}:
            anchor_completed, anchor_failed = run_items(anchor_items)
            completed += anchor_completed
            failed += anchor_failed
        elif anchor_items:
            raise SystemExit(
                "RESIDUAL JUDGING BLOCKED: the 50-anchor stage is incomplete"
            )
        historical_rows = read_jsonl(resolve_cfg_path(cfg, "utility_registry"))
        _, historical_registry = complete_utility_v2_rows(historical_rows)
        current_anchor_rows = read_jsonl(valid_paths["anchor"])
        stability = anchor_report(historical_registry, current_anchor_rows)
        write_json(complete_dir / "anchor_stability_report.json", stability)
        if stability["verdict"] not in {"STABLE", "STABLE_WITH_MINOR_DRIFT"}:
            raise SystemExit(
                "RESIDUAL JUDGING BLOCKED BY ANCHOR GATE: "
                f"{stability['verdict']}"
            )
        if args.stage == "anchors":
            write_json(complete_dir / "anchor_gate_execution_manifest.json", {
                "schema": "utility-v2-anchor-gate-execution-v1",
                "batch_id": batch_id,
                "stage": "anchors",
                "anchor_gate_verdict": stability["verdict"],
                "anchors_valid": len(current_anchor_rows),
                "completed_this_invocation": completed,
                "failed_this_invocation": failed,
                "residual_requests_made": 0,
                "external_disclosure_user_approved": True,
                "test_read": False,
                "created_utc": utc_now(),
            })
        else:
            residual_completed, residual_failed = run_items(residual_items)
            completed += residual_completed
            failed += residual_failed
    finally:
        canonical_judge.call_chat = original_call_chat
        for handle in handles.values():
            handle.close()

    if args.stage == "anchors":
        print(json.dumps({
            "completed": completed,
            "failed": failed,
            "anchor_gate": stability["verdict"],
            "anchors_valid": len(current_anchor_rows),
            "residual_requests_made": 0,
            "output_dir": str(complete_dir),
        }, ensure_ascii=False, indent=2))
        return

    residual_valid = read_jsonl(valid_paths["residual"])
    anchor_valid = read_jsonl(valid_paths["anchor"])
    failure_rows = read_jsonl(failure_path)
    write_jsonl(
        complete_dir / "rejected_or_invalid_judgments.jsonl",
        failure_rows,
    )
    if len(anchor_valid) != int(cfg["expected_counts"]["anchors"]):
        raise SystemExit(
            f"MERGE BLOCKED: expected 50 valid anchors, got {len(anchor_valid)}"
        )
    expected_residual = int(cfg["expected_counts"]["final_unique_pending"])
    if len(residual_valid) != expected_residual or failure_rows:
        raise SystemExit(
            "MERGE BLOCKED: residual judgments are not coverage-complete "
            f"({len(residual_valid)}/{expected_residual} valid; "
            f"{len(failure_rows)} failures)"
        )

    historical_path = resolve_cfg_path(cfg, "utility_registry")
    historical_rows = read_jsonl(historical_path)
    _, historical_registry = complete_utility_v2_rows(historical_rows)
    union_by_pair = {
        (str(row["query_id"]), str(row["comment_id"])): row for row in union
    }
    new_registry_rows = []
    for row in sorted(
        residual_valid,
        key=lambda item: (str(item["query_id"]), str(item["comment_id"])),
    ):
        key = (str(row["query_id"]), str(row["comment_id"]))
        if key in historical_registry:
            raise SystemExit(f"MERGE BLOCKED: would overwrite historical row {key}")
        scores = {dim: int(row["validated_scores"][dim]) for dim in DIMS_V2}
        linear = (
            0.25 * scores["relevance"]
            + 0.30 * scores["usefulness"]
            + 0.15 * scores["actionability"]
            + 0.10 * scores["novelty"]
            + 0.10 * scores["resonance"]
            + 0.10 * scores["safety"]
        )
        expected_utility = min(linear, 2.0) if scores["safety"] <= 2 else linear
        if abs(float(row["utility"]) - round(expected_utility, 4)) > 1e-9:
            raise SystemExit(f"MERGE BLOCKED: utility mismatch for {key}")
        audit = audit_paths.get(key)
        if audit is None:
            # A resume can use already-complete validated rows.  Recover the
            # deterministic paths and verify they exist rather than guessing.
            admin = union_by_pair[key]
            stem = (
                f"residual_{int(admin['payload_index']):04d}_"
                f"{admin['provider_payload_sha256'][:12]}"
            )
            request_path = request_dir / f"{stem}.json"
            response_path = response_dir / f"{stem}.json"
            if not request_path.exists() or not response_path.exists():
                raise SystemExit(f"MERGE BLOCKED: missing audit provenance for {key}")
            response_audit = read_json_object(response_path)
            audit = {
                "raw_request_path": str(request_path.relative_to(root)),
                "raw_response_path": str(response_path.relative_to(root)),
                "provider_payload_sha256": admin["provider_payload_sha256"],
                "request_body_sha256": read_json_object(request_path)[
                    "request_body_sha256"
                ],
                "completed_utc": response_audit["completed_utc"],
            }
        meta = union_by_pair[key]
        default_registry_metadata_fields = ("residual_sources", "source_pools")
        registry_metadata_fields = cfg.get(
            "registry_metadata_fields",
            default_registry_metadata_fields,
        )
        registry_metadata = {
            str(field): meta[str(field)]
            for field in registry_metadata_fields
            if str(field) in meta
        }
        new_registry_rows.append({
            "query_id": key[0],
            "comment_id": key[1],
            **{f"label_{dim}": scores[dim] for dim in DIMS_V2},
            "utility": round(expected_utility, 4),
            "linear_utility_before_safety_gate": round(linear, 4),
            "safety_gate_applied": scores["safety"] <= 2,
            "rationale": row.get("rationale", ""),
            "judge_model": row["model"],
            "judge_id": "utility-v2",
            "judge_version": row["judge_version"],
            "batch_id": batch_id,
            "label_role": "LLM simulated-user silver; not human gold",
            "judgment_source": cfg.get(
                "judgment_source",
                "coverage_complete_drift_residual",
            ),
            **registry_metadata,
            "prompt_sha256": row["prompt_sha256"],
            "rendered_payload_prompt_sha256": row[
                "rendered_payload_prompt_sha256"
            ],
            "provider_payload_sha256": audit["provider_payload_sha256"],
            "raw_request_path": audit["raw_request_path"],
            "raw_response_path": audit["raw_response_path"],
            "validation_status": row["validation_status"],
            "validation_timestamp_utc": audit["completed_utc"],
        })

    write_jsonl(
        complete_dir / "new_validated_judgments.jsonl",
        new_registry_rows,
    )
    write_jsonl(
        complete_dir / "registry_delta.jsonl",
        [{"operation": "ADD", "row": row} for row in new_registry_rows],
    )
    coverage_registry_path = (
        complete_dir / "utility_registry_coverage_complete.jsonl"
    )
    with historical_path.open("rb") as source, coverage_registry_path.open("xb") as dest:
        shutil.copyfileobj(source, dest)
        source.seek(0, 2)
        if source.tell():
            source.seek(-1, 2)
            if source.read(1) != b"\n":
                dest.write(b"\n")
        for row in new_registry_rows:
            dest.write(
                (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode(
                    "utf-8"
                )
            )
    merged_rows = read_jsonl(coverage_registry_path)
    merged_complete, merged_registry = complete_utility_v2_rows(merged_rows)
    if len(merged_complete) != len(historical_registry) + expected_residual:
        raise SystemExit("MERGE BLOCKED: complete registry row count mismatch")
    if not all(key in merged_registry for key in union_by_pair):
        raise SystemExit("MERGE BLOCKED: one or more residual pairs absent after merge")

    execution_manifest = {
        "batch_id": batch_id,
        "completed_this_invocation": completed,
        "failed_this_invocation": failed,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "external_disclosure_user_approved": True,
        "anchor_gate_ran_before_residual": True,
        "anchor_gate_verdict": stability["verdict"],
        "anchors_valid": len(anchor_valid),
        "residuals_valid": len(residual_valid),
        "historical_complete_rows": len(historical_registry),
        "new_rows_added": len(new_registry_rows),
        "historical_rows_overwritten": 0,
        "coverage_complete_registry_rows": len(merged_complete),
        "provider_payload_identity_fields": 0,
        "raw_request_files": len(list(request_dir.glob("*.json"))),
        "raw_response_files": len(list(response_dir.glob("*.json"))),
        "registry_merge_executed": True,
        "test_read": False,
        "created_utc": utc_now(),
    }
    write_json(complete_dir / "judge_execution_manifest.json", execution_manifest)
    output_names = (
        "anchor_stability_report.json",
        "new_validated_judgments.jsonl",
        "rejected_or_invalid_judgments.jsonl",
        "registry_delta.jsonl",
        "utility_registry_coverage_complete.jsonl",
        "judge_execution_manifest.json",
    )
    write_json(complete_dir / "judging_manifest.json", {
        "schema": "coverage-complete-residual-judging-manifest-v1",
        **execution_manifest,
        "preflight_manifest_sha256": sha256(
            preflight_dir / "preflight_manifest.json"
        ),
        "preflight_manifest_canonical_sha256": hash_json(manifest),
        "protocol_lock_sha256": sha256(
            preflight_dir / "judge_protocol_lock.json"
        ),
        "historical_registry_sha256": sha256(historical_path),
        "coverage_complete_registry_sha256": sha256(coverage_registry_path),
        "output_hashes": {
            name: sha256(complete_dir / name) for name in output_names
        },
    })
    print(json.dumps({
        "completed": completed,
        "failed": failed,
        "anchor_gate": stability["verdict"],
        "new_registry_rows": len(new_registry_rows),
        "output_dir": str(complete_dir),
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "judge"))
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--config", type=Path, default=root / "configuration/params.yaml"
    )
    parser.add_argument("--config-key", default=CONFIG_KEY)
    parser.add_argument("--allow-external-judging", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("anchors", "residual", "all"),
        default="all",
        help=(
            "Run only the 50-anchor gate, only the residual continuation after "
            "a persisted passing gate, or both in strict order."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preflight":
        run_preflight(args)
    else:
        run_authorized_judge(args)


if __name__ == "__main__":
    main()
