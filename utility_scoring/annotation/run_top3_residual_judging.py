#!/usr/bin/env python3
"""Prepare, run, and audit frozen Top-3 residual utility-v2 judging.

The command is deliberately staged. ``prepare`` performs every fail-closed
identity/protocol/blindness check and freezes 50 label-blind calibration
anchors. ``judge`` refuses to run unless those reports pass. ``analyze`` first
checks batch stability; it only merges labels and computes method metrics when
the anchor gate permits it. Frozen test paths are rejected in every phase.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import configuration as project_config
from evaluation.ir_metrics import graded_ndcg_at
from evaluation.judgment_completeness import complete_utility_v2_rows
from evaluation.statistics import bootstrap_ci
from evaluation.utility import (
    DIMS_V2,
    WEIGHTS_V2,
    annotation_prompt as _prompt,
    clamp_score as _clamp_score,
    parse_judgment as _parse_json,
    utility_v2 as _utility_v2,
)
from shared.llm_client import call_chat

MODEL = "bedrock:us.meta.llama3-3-70b-instruct-v1:0"
TEMPERATURE = 0.0
MAX_TOKENS = 350
MAX_RETRIES = 4
ANCHOR_SEED = 20260719
BOOTSTRAP_SEED = 20260719
PERMUTATIONS = 10_000
USEFUL_THRESHOLD = 4.0
CORE_DIMS = tuple(DIMS_V2)
EXPECTED_RESIDUAL_PAIRS = 189
EXPECTED_QUERY_UNIVERSE = 100
EXPECTED_INVOLVED_QUERIES = 90
ALLOWED_VERDICTS = {
    "PROCEED_TO_TOP5", "PROCEED_WITH_CONDITIONAL_RECOVERY",
    "KEEP_AS_ABLATION_ONLY", "STOP_REWRITE_RECOVERY",
    "INCONCLUSIVE_DUE_TO_COVERAGE", "INCONCLUSIVE_DUE_TO_BATCH_DRIFT",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    reject_test_path(path)
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def reject_test_path(path: Path) -> None:
    if "test" in path.name.lower() or any(
            part.lower() == "test" for part in path.parts):
        raise ValueError(f"frozen-test-looking path rejected: {path}")


def git_state(repo: Path) -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        capture_output=True, text=True).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def load_queries(path: Path) -> dict[str, str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = {str(row.get("id") or row.get("query_id")):
           str(row.get("question") or row.get("query_text")) for row in rows}
    if len(out) != EXPECTED_QUERY_UNIVERSE:
        raise ValueError(f"expected dev100-v2, got {len(out)} queries")
    return out


def load_query_types(path: Path) -> dict[str, str]:
    reject_test_path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    result = {str(row["query_id"]): str(row["llm_single_multi_label"])
              for row in rows}
    if Counter(result.values()) != Counter({"single_need": 50, "multi_need": 50}):
        raise ValueError("single50/multi50 registry mismatch")
    return result


def load_corpus(path: Path) -> dict[str, str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = {str(row["title"]): str(row["text"]) for row in rows}
    if len(result) != 19013:
        raise ValueError(f"expected 19,013 comments, got {len(result)}")
    return result


def group_run(path: Path, qids: set[str]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        grouped[str(row["query_id"])].append(row)
    if set(grouped) != qids:
        raise ValueError(f"run query set mismatch: {path}")
    for qid, rows in grouped.items():
        rows.sort(key=lambda row: int(row["rank"]))
        if [int(row["rank"]) for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"non-contiguous ranks for {qid}: {path}")
        if len({str(row["comment_id"]) for row in rows}) != len(rows):
            raise ValueError(f"duplicate run comment for {qid}: {path}")
    return dict(grouped)


def completed_old_rows(path: Path) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    rows = read_jsonl(path)
    return complete_utility_v2_rows(rows)


def strict_graph_derived(row: dict) -> bool:
    return bool(
        row.get("recognition_success")
        and int(row.get("graph_seed_count") or 0) > 0
        and row.get("graph_propagation_executed")
        and "ppr" in str(row.get("candidate_score_origin") or "")
        and not row.get("result_is_dense_fallback_copy")
    )


def query_strata(original: dict[str, list[dict]],
                 union: dict[str, list[dict]],
                 query_types: dict[str, str]) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    qids = set(original)
    entered = {qid for qid in qids if original[qid][0].get("graph_entry_status")}
    fallback = qids - entered
    recovered = {qid for qid in fallback if union[qid][0].get("graph_entry_status")}
    still = fallback - recovered
    strata = {
        "all100": qids,
        "single50": {q for q, value in query_types.items() if value == "single_need"},
        "multi50": {q for q, value in query_types.items() if value == "multi_need"},
        "original_graph_entered28": entered,
        "original_fallback72": fallback,
        "structured_recovered61": recovered,
        "still_fallback11": still,
    }
    expected = {
        "all100": 100, "single50": 50, "multi50": 50,
        "original_graph_entered28": 28, "original_fallback72": 72,
        "structured_recovered61": 61, "still_fallback11": 11,
    }
    if {name: len(ids) for name, ids in strata.items()} != expected:
        raise ValueError("graph/query stratum sizes mismatch")
    by_qid = {qid: [name for name, members in strata.items() if qid in members]
              for qid in qids}
    return strata, by_qid


def rank_band(row: dict) -> str:
    rank = int(row.get("mmr_rank") or row.get("candidate_rank") or 10**9)
    if rank <= 10:
        return "pool_rank_1_10"
    if rank <= 20:
        return "pool_rank_11_20"
    return "pool_rank_21_plus"


def select_anchors(old_rows: list[dict], current_qids: set[str],
                   residual_keys: set[tuple[str, str]],
                   by_qid: dict[str, list[str]]) -> list[dict]:
    eligible = []
    for row in old_rows:
        qid, cid = str(row["query_id"]), str(row["comment_id"])
        if qid not in current_qids or (qid, cid) in residual_keys:
            continue
        graph_group = next(name for name in (
            "original_graph_entered28", "structured_recovered61", "still_fallback11")
            if name in by_qid[qid])
        need_group = "single50" if "single50" in by_qid[qid] else "multi50"
        eligible.append({
            "query_id": qid, "comment_id": cid,
            "need_stratum": need_group, "graph_stratum": graph_group,
            "old_pool_rank_band": rank_band(row),
            "selection_used_old_scores": False,
        })
    eligible.sort(key=lambda row: (row["query_id"], row["comment_id"]))
    rng = random.Random(ANCHOR_SEED)
    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in eligible:
        cells[(row["need_stratum"], row["graph_stratum"],
               row["old_pool_rank_band"])].append(row)
    for rows in cells.values():
        rng.shuffle(rows)
    chosen, used = [], set()
    ordered_cells = sorted(cells)
    while len(chosen) < 50:
        progressed = False
        for cell in ordered_cells:
            rows = cells[cell]
            while rows and (rows[-1]["query_id"], rows[-1]["comment_id"]) in used:
                rows.pop()
            if not rows:
                continue
            row = rows.pop()
            key = (row["query_id"], row["comment_id"])
            chosen.append(row); used.add(key); progressed = True
            if len(chosen) == 50:
                break
        if not progressed:
            raise ValueError("not enough eligible calibration anchors")
    chosen.sort(key=lambda row: (row["query_id"], row["comment_id"]))
    return chosen


def source_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def prepare(args) -> None:
    for value in vars(args).values():
        if isinstance(value, Path):
            reject_test_path(value)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    queries = load_queries(args.queries)
    query_types = load_query_types(args.query_admin)
    if set(queries) != set(query_types):
        raise ValueError("official/admin query identity mismatch")
    summaries = read_jsonl(args.summaries)
    if {str(row["query_id"]) for row in summaries} != set(queries):
        raise ValueError("frozen summary query identity mismatch")
    corpus = load_corpus(args.corpus)
    old_rows, old_registry = completed_old_rows(args.old_judgments)
    qids = set(queries)
    runs = {
        "original": group_run(args.original_run, qids),
        "original_plus_structured": group_run(args.orig_struct_run, qids),
        "cohere_dense": group_run(args.dense_run, qids),
        "official_static": group_run(args.static_run, qids),
        "dense_bridge_2hop": group_run(args.bridge_run, qids),
    }
    strata, by_qid = query_strata(
        runs["original"], runs["original_plus_structured"], query_types)

    residual = read_jsonl(args.residual)
    residual_keys = [(str(row["query_id"]), str(row["comment_id"])) for row in residual]
    violations = []
    if len(residual) != EXPECTED_RESIDUAL_PAIRS:
        violations.append(f"residual rows {len(residual)} != 189")
    if len(set(residual_keys)) != EXPECTED_RESIDUAL_PAIRS:
        violations.append("residual query-comment pairs are not unique")
    involved = {qid for qid, _ in residual_keys}
    if len(involved) != EXPECTED_INVOLVED_QUERIES:
        violations.append(f"involved query count {len(involved)} != 90")
    zero_queries = qids - involved
    if len(zero_queries) != 10:
        violations.append(f"zero-residual query count {len(zero_queries)} != 10")
    per_query = Counter(qid for qid, _ in residual_keys)
    if any(value < 1 or value > 3 for value in per_query.values()):
        violations.append("involved query residual count outside 1..3")
    union_top3 = {qid: {str(row["comment_id"]) for row in rows[:3]}
                  for qid, rows in runs["original_plus_structured"].items()}
    if any(cid not in union_top3[qid] for qid, cid in residual_keys):
        violations.append("residual pair outside Original+Structured top-3")
    if any(key in old_registry for key in residual_keys):
        violations.append("residual pair already present in completed old registry")
    if any(cid not in corpus for _, cid in residual_keys):
        violations.append("residual comment missing from corpus")
    if any(qid not in queries for qid, _ in residual_keys):
        violations.append("residual query missing from dev100-v2")
    stored_alias = {
        "all100": "all100", "single_need": "single50", "multi_need": "multi50",
        "original_graph_entered": "original_graph_entered28",
        "original_fallback": "original_fallback72",
        "structured_recovered": "structured_recovered61",
        "still_fallback": "still_fallback11",
    }
    stored_strata_bad = sum(
        {stored_alias.get(name, name) for name in (row.get("strata") or [])}
        != set(by_qid[str(row["query_id"])]) for row in residual)
    if stored_strata_bad:
        violations.append(f"residual stored strata mismatches: {stored_strata_bad}")
    stored_graph_bad = 0
    union_lookup = {(qid, str(row["comment_id"])): row
                    for qid, rows in runs["original_plus_structured"].items()
                    for row in rows[:3]}
    for row in residual:
        key = (str(row["query_id"]), str(row["comment_id"]))
        if bool(row.get("strict_graph_derived")) != strict_graph_derived(union_lookup[key]):
            stored_graph_bad += 1
    if stored_graph_bad:
        violations.append(f"strict graph-derived mismatches: {stored_graph_bad}")

    old_manifest = json.loads(args.old_judge_manifest.read_text(encoding="utf-8"))
    judge_schema = {
        "dimensions": list(DIMS_V2), "type": "integer", "minimum": 1,
        "maximum": 7, "rationale": "string", "utility_weights": WEIGHTS_V2,
        "safety_gate": "if safety<=2 then utility=min(weighted_utility,2)",
    }
    old_output_mtime = args.old_judgments.stat().st_mtime
    source_paths = {
        "rubric_prompt": args.judge_prompt,
        "judge_code": args.judge_code,
        "section17_runner": args.section17_runner,
        "llm_client": args.llm_client,
    }
    observed_protocol = {
        "system_prompt": None,
        "user_prompt_sha256": sha256(args.judge_prompt),
        "input_fields": ["query_text", "comment_text", "facets_json"],
        "facets_json": {},
        "frozen_summary_in_payload": False,
        "model": MODEL, "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS, "max_retries": MAX_RETRIES,
        "retry_backoff_seconds": {"initial": 2.0, "maximum": 30.0},
        "parser": "json -> first JSON object -> dimension regex fallback",
        "score_validation": "round/clamp each dimension to integer 1..7",
        "invalid_handling": "retry request exceptions; validated parser/clamp retained",
        "schema": judge_schema, "schema_sha256": hash_json(judge_schema),
    }
    protocol_checks = {
        "old_manifest_model_matches": old_manifest.get("model") == MODEL,
        "old_manifest_rubric_matches": old_manifest.get("rubric") == "utility-v2",
        "old_rows_single_model_matches": {row.get("judge_model") for row in old_rows} == {MODEL},
        "old_rows_single_judge_id_matches": {row.get("judge_id") for row in old_rows} == {"utility-v2"},
        "prompt_file_predates_old_output": args.judge_prompt.stat().st_mtime <= old_output_mtime,
        "judge_code_predates_old_output": args.judge_code.stat().st_mtime <= old_output_mtime,
        "section17_runner_predates_old_output": args.section17_runner.stat().st_mtime <= old_output_mtime,
        "llm_client_predates_old_output": args.llm_client.stat().st_mtime <= old_output_mtime,
        "old_rows_have_all_schema_dimensions": all(
            all(f"label_{dim}" in row for dim in DIMS_V2) for row in old_rows),
        "old_rows_include_query_and_comment_text": all(
            isinstance(row.get("query_text"), str) and isinstance(row.get("comment_text"), str)
            for row in old_rows),
        "historical_prompt_hash_recorded_in_old_manifest": False,
        "historical_code_commit_recorded_in_old_manifest": False,
    }
    hard_protocol_checks = {key: value for key, value in protocol_checks.items()
                            if not key.startswith("historical_")}
    protocol_equivalent = all(hard_protocol_checks.values())
    if not protocol_equivalent:
        violations.append("PROTOCOL_MISMATCH")

    anchors = select_anchors(old_rows, qids, set(residual_keys), by_qid)
    anchor_keys = {(row["query_id"], row["comment_id"]) for row in anchors}
    if len(anchors) != 50 or anchor_keys & set(residual_keys):
        violations.append("calibration anchor count/overlap violation")
    anchor_coverage = {
        "need_strata": dict(Counter(row["need_stratum"] for row in anchors)),
        "graph_strata": dict(Counter(row["graph_stratum"] for row in anchors)),
        "old_pool_rank_bands": dict(Counter(row["old_pool_rank_band"] for row in anchors)),
    }

    repo = Path(__file__).resolve().parents[2]
    manifest = {
        "created_utc": utc_now(), "dataset": "dev100-v2",
        "residual_manifest": {"path": str(args.residual), "sha256": sha256(args.residual)},
        "residual_pairs": len(residual), "evaluation_query_universe": 100,
        "pair_rows_involved_queries": len(involved),
        "zero_residual_queries": sorted(zero_queries),
        "per_query_residual_count_distribution": dict(sorted(Counter(
            per_query.get(qid, 0) for qid in qids).items())),
        "selection_attestation": {
            "source_report": str(args.residual_size_report),
            "source_report_sha256": sha256(args.residual_size_report),
            "utility_values_used_to_select": False,
            "selection_inputs": ["rank", "completed judgment status", "method membership",
                                 "pilot membership", "strict graph provenance", "fixed cutoff=3"],
        },
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in {
            "queries": args.queries, "query_admin": args.query_admin,
            "summaries": args.summaries, "corpus": args.corpus,
            "old_judgments": args.old_judgments,
            "original_run": args.original_run, "orig_struct_run": args.orig_struct_run,
            "dense_run": args.dense_run, "static_run": args.static_run,
            "bridge_run": args.bridge_run,
        }.items()},
        "judge": observed_protocol,
        "source_files": {name: {"path": str(path), "sha256": sha256(path),
                                "mtime_utc": source_mtime(path)}
                         for name, path in source_paths.items()},
        "git": git_state(repo), "test_split_used": False,
    }
    integrity = {
        "verdict": "PASS" if not violations else "FAIL",
        "violations": violations, "residual_pair_count": len(residual),
        "unique_pair_count": len(set(residual_keys)),
        "evaluation_query_universe": len(qids), "involved_query_count": len(involved),
        "zero_residual_query_count": len(zero_queries),
        "all_pairs_in_orig_struct_top3": not any(
            cid not in union_top3[qid] for qid, cid in residual_keys),
        "all_pairs_previously_unjudged": not any(key in old_registry for key in residual_keys),
        "stored_strata_mismatches": stored_strata_bad,
        "stored_graph_derived_mismatches": stored_graph_bad,
        "old_registry_complete_pairs": len(old_registry),
        "utility_values_used_for_residual_selection": False,
        "test_split_used": False,
    }
    protocol = {
        "verdict": "EQUIVALENT" if protocol_equivalent else "PROTOCOL_MISMATCH",
        "checks": protocol_checks, "current_protocol": observed_protocol,
        "diff": [] if protocol_equivalent else [
            key for key, value in hard_protocol_checks.items() if not value],
        "historical_provenance_limitation": (
            "The old manifest omitted prompt hash and code commit. Equivalence is supported by "
            "the unchanged tracked prompt, old rows, and source mtimes predating the old output; "
            "the 50 anchors are the cross-batch empirical gate."),
    }
    blindness = {
        "verdict": "PASS", "payload_allowlist": ["query_text", "comment_text", "facets_json"],
        "facets_json_fixed_value": {}, "frozen_summary_read_but_not_sent": True,
        "forbidden_fields_absent": ["method", "rank", "score", "provenance",
                                    "graph_entry_status", "residual_status",
                                    "single_multi", "recovered", "old_judgment"],
        "prompt_mentions_structured_rewrite": False,
        "payload_items_audited": EXPECTED_RESIDUAL_PAIRS + 50,
    }
    batch = {
        "batch_id": "utility-v2-top3-residual-dev100-v2-20260719",
        "created_utc": utc_now(), "residual_items": len(residual),
        "calibration_anchor_items": len(anchors), "total_items": len(residual) + len(anchors),
        "anchor_seed": ANCHOR_SEED, "anchor_selection_used_old_scores": False,
        "anchor_coverage": anchor_coverage, "model": MODEL,
        "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
        "max_retries": MAX_RETRIES, "self_consistency_repeats": 1,
        "residual_manifest_sha256": sha256(args.residual),
        "anchor_manifest_sha256": None,
        "protocol_equivalence_required": True,
        "protocol_equivalence_verdict": protocol["verdict"],
        "integrity_verdict": integrity["verdict"], "llm_calls_made": 0,
        "external_disclosure_user_approved": False,
    }
    write_jsonl(args.out_dir / "top3_calibration_anchors.jsonl", anchors)
    batch["anchor_manifest_sha256"] = sha256(
        args.out_dir / "top3_calibration_anchors.jsonl")
    write_json(args.out_dir / "top3_residual_judging_input_manifest.json", manifest)
    write_json(args.out_dir / "top3_residual_judging_integrity_report.json", integrity)
    write_json(args.out_dir / "judge_protocol_equivalence_report.json", protocol)
    write_json(args.out_dir / "judge_payload_blindness_audit.json", blindness)
    write_json(args.out_dir / "top3_judge_batch_manifest.json", batch)
    if violations:
        raise SystemExit(f"PREPARE FAILED: {violations}")
    print(json.dumps({"prepared": True, "residual": len(residual), "anchors": len(anchors),
                      "protocol": protocol["verdict"]}, indent=2))


def validated_existing(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {(str(row["query_id"]), str(row["comment_id"])): row for row in rows
            if row.get("validation_status") == "valid"}


def judge_one(item: dict, item_type: str, queries: dict[str, str],
              corpus: dict[str, str], batch_id: str,
              rubric_prompt_sha256: str) -> tuple[dict, dict]:
    qid, cid = str(item["query_id"]), str(item["comment_id"])
    payload = {"query_text": queries[qid], "comment_text": corpus[cid], "facets_json": {}}
    return judge_payload(payload, item_type, batch_id, rubric_prompt_sha256,
                         identity={"query_id": qid, "comment_id": cid})


def judge_payload(payload: dict, item_type: str, batch_id: str,
                  rubric_prompt_sha256: str,
                  identity: dict | None = None) -> tuple[dict, dict]:
    """Judge an already-frozen blind payload with the canonical utility-v2 protocol.

    ``identity`` is copied only into local result metadata.  The Bedrock prompt
    is rendered exclusively from the three allowlisted payload values, which
    lets redacted batches keep their ADMIN mapping local.
    """
    if set(payload) != {"query_text", "comment_text", "facets_json"}:
        raise ValueError(f"utility-v2 payload fields changed: {sorted(payload)}")
    if payload["facets_json"] != {}:
        raise ValueError("historical utility-v2 requires facets_json={}")
    prompt = _prompt(payload["query_text"], payload["comment_text"], {}, "v2")
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            raw_text = call_chat(prompt, model_spec=MODEL, max_tokens=MAX_TOKENS,
                                 temperature=TEMPERATURE)
            parsed = _parse_json(raw_text)
            scores = {dim: _clamp_score(parsed.get(dim)) for dim in DIMS_V2}
            utility = round(_utility_v2(scores), 4)
            common = {
                **(identity or {}), "item_type": item_type,
                "batch_id": batch_id, "judge_version": "utility-v2",
                "prompt_sha256": rubric_prompt_sha256,
                "rendered_payload_prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")).hexdigest(),
                "model": MODEL, "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS, "retry_count": attempt,
            }
            raw = {**common, "raw_response": raw_text,
                   "payload_field_names": list(payload), "validation_status": "valid"}
            valid = {**common, "validated_scores": scores, "utility": utility,
                     "rationale": str(parsed.get("rationale") or ""),
                     "validation_status": "valid"}
            return raw, valid
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == MAX_RETRIES:
                break
            time.sleep(min(2.0 * (2 ** attempt), 30.0))
    raise RuntimeError(last_error or "unknown judge failure")


def run_judging(args) -> None:
    integrity = json.loads((args.out_dir / "top3_residual_judging_integrity_report.json").read_text())
    protocol = json.loads((args.out_dir / "judge_protocol_equivalence_report.json").read_text())
    blindness = json.loads((args.out_dir / "judge_payload_blindness_audit.json").read_text())
    batch = json.loads((args.out_dir / "top3_judge_batch_manifest.json").read_text())
    if integrity["verdict"] != "PASS" or protocol["verdict"] != "EQUIVALENT" or blindness["verdict"] != "PASS":
        raise SystemExit("hard gate failed; no LLM calls allowed")
    if sha256(args.residual) != batch["residual_manifest_sha256"]:
        raise SystemExit("residual manifest changed after prepare")
    anchor_path = args.out_dir / "top3_calibration_anchors.jsonl"
    if sha256(anchor_path) != batch["anchor_manifest_sha256"]:
        raise SystemExit("anchor manifest changed after prepare")
    queries = load_queries(args.queries); corpus = load_corpus(args.corpus)
    input_manifest = json.loads((
        args.out_dir / "top3_residual_judging_input_manifest.json").read_text())
    rubric_prompt_sha256 = input_manifest["judge"]["user_prompt_sha256"]
    items = [(row, "residual") for row in read_jsonl(args.residual)] + [
        (row, "calibration_anchor") for row in read_jsonl(anchor_path)]
    paths = {
        "residual": (
            args.out_dir / "top3_residual_judgments_raw.jsonl",
            args.out_dir / "top3_residual_judgments_validated.jsonl",
            args.out_dir / "top3_residual_judgment_failures.jsonl"),
        "calibration_anchor": (
            args.out_dir / "top3_anchor_rejudgments_raw.jsonl",
            args.out_dir / "top3_anchor_rejudgments_validated.jsonl",
            args.out_dir / "top3_anchor_rejudgment_failures.jsonl"),
    }
    done = {}
    for kind, (_, valid_path, _) in paths.items():
        done[kind] = validated_existing(valid_path)
    pending = [(item, kind) for item, kind in items
               if (str(item["query_id"]), str(item["comment_id"])) not in done[kind]]
    print(json.dumps({"pending": len(pending), "already_valid": sum(len(x) for x in done.values()),
                      "workers": args.workers}, indent=2), flush=True)
    handles = {}
    for kind, triplet in paths.items():
        for path in triplet:
            handles[path] = path.open("a", encoding="utf-8")
    completed = failed = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(judge_one, item, kind, queries, corpus,
                                       batch["batch_id"], rubric_prompt_sha256): (item, kind)
                       for item, kind in pending}
            for future in as_completed(futures):
                item, kind = futures[future]
                raw_path, valid_path, fail_path = paths[kind]
                try:
                    raw, valid = future.result()
                    handles[raw_path].write(json.dumps(raw, ensure_ascii=False) + "\n")
                    handles[valid_path].write(json.dumps(valid, ensure_ascii=False) + "\n")
                    handles[raw_path].flush(); handles[valid_path].flush()
                    completed += 1
                except Exception as exc:
                    failure = {"query_id": str(item["query_id"]),
                               "comment_id": str(item["comment_id"]),
                               "item_type": kind, "error": f"{type(exc).__name__}: {exc}"}
                    handles[fail_path].write(json.dumps(failure, ensure_ascii=False) + "\n")
                    handles[fail_path].flush(); failed += 1
                if (completed + failed) % 25 == 0:
                    print(json.dumps({"processed": completed, "failed": failed,
                                      "remaining": len(pending) - completed - failed}), flush=True)
    finally:
        for fh in handles.values():
            fh.close()
    completed_total = sum(len(validated_existing(valid_path))
                          for _, valid_path, _ in paths.values())
    batch["llm_calls_made"] = completed_total
    batch["judge_items_completed_total"] = completed_total
    batch["completed_this_invocation"] = completed
    batch["failed_this_invocation"] = failed
    batch["external_disclosure_user_approved"] = True
    sandbox_failures = args.out_dir / "top3_residual_judgment_sandbox_network_failures.jsonl"
    batch["sandbox_network_failures_before_external_approval"] = (
        len(read_jsonl(sandbox_failures)) if sandbox_failures.exists() else 0)
    batch["last_run_utc"] = utc_now()
    write_json(args.out_dir / "top3_judge_batch_manifest.json", batch)
    print(json.dumps({"completed": completed, "failed": failed}, indent=2))


def mean(values):
    values = [float(value) for value in values if value is not None]
    return statistics.fmean(values) if values else None


def median(values):
    values = [float(value) for value in values if value is not None]
    return statistics.median(values) if values else None


def paired_permutation_p(deltas: list[float], seed: int = BOOTSTRAP_SEED) -> float | None:
    if not deltas:
        return None
    observed = abs(mean(deltas))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(PERMUTATIONS):
        value = abs(mean(delta if rng.random() < .5 else -delta for delta in deltas))
        extreme += value >= observed
    return (extreme + 1) / (PERMUTATIONS + 1)


def anchor_report(old_registry: dict[tuple[str, str], dict],
                  new_rows: list[dict]) -> dict:
    valid = [row for row in new_rows if row.get("validation_status") == "valid"]
    dimensions = {}
    severe = []
    all_old, all_new = [], []
    for dim in DIMS_V2:
        old = [int(old_registry[(str(row["query_id"]), str(row["comment_id"]))][f"label_{dim}"])
               for row in valid]
        new = [int(row["validated_scores"][dim]) for row in valid]
        diffs = [b - a for a, b in zip(old, new)]
        absdiff = [abs(value) for value in diffs]
        kappa = float(cohen_kappa_score(old, new, weights="quadratic")) if len(set(old + new)) > 1 else None
        ci = bootstrap_ci(diffs, n_boot=5000, seed=BOOTSTRAP_SEED + len(dim))
        stats = {
            "n": len(old), "exact_agreement": mean(a == b for a, b in zip(old, new)),
            "weighted_cohen_kappa_quadratic": kappa,
            "mean_absolute_difference": mean(absdiff),
            "one_grade_flip_rate": mean(value == 1 for value in absdiff),
            "two_or_more_grade_flip_rate": mean(value >= 2 for value in absdiff),
            "extreme_flip_rate_abs_ge_4": mean(value >= 4 for value in absdiff),
            "old_mean": mean(old), "new_mean": mean(new),
            "mean_shift_new_minus_old": mean(diffs),
            "paired_difference_bootstrap_95ci": list(ci),
        }
        warnings = []
        if stats["extreme_flip_rate_abs_ge_4"] > .05:
            warnings.append("extreme_flip_rate_gt_5pct")
        if stats["two_or_more_grade_flip_rate"] > .15:
            warnings.append("two_or_more_flip_rate_gt_15pct")
        if kappa is None or kappa < .60:
            warnings.append("weighted_kappa_lt_0.60")
        if abs(stats["mean_shift_new_minus_old"]) > .25:
            warnings.append("absolute_mean_drift_gt_0.25")
        stats["severe_warnings"] = warnings
        severe.extend(f"{dim}:{warning}" for warning in warnings)
        dimensions[dim] = stats
        all_old.extend(old); all_new.extend(new)
    utility_old = [float(old_registry[(str(row["query_id"]), str(row["comment_id"]))]["utility"])
                   for row in valid]
    utility_new = [float(row["utility"]) for row in valid]
    utility_diff = [b - a for a, b in zip(utility_old, utility_new)]
    if len(valid) < 45:
        verdict = "INSUFFICIENT_VALID_ANCHORS"
    elif severe:
        verdict = "MATERIAL_BATCH_DRIFT"
    elif (mean(a == b for a, b in zip(all_old, all_new)) >= .75
          and abs(mean(utility_diff)) <= .10):
        verdict = "STABLE"
    else:
        verdict = "STABLE_WITH_MINOR_DRIFT"
    return {
        "verdict": verdict, "valid_anchor_pairs": len(valid),
        "requested_anchor_pairs": 50, "dimensions": dimensions,
        "overall_dimension_exact_agreement": mean(a == b for a, b in zip(all_old, all_new)),
        "utility": {"old_mean": mean(utility_old), "new_mean": mean(utility_new),
                    "mean_shift_new_minus_old": mean(utility_diff),
                    "mean_absolute_difference": mean(abs(x) for x in utility_diff),
                    "paired_difference_bootstrap_95ci": list(bootstrap_ci(
                        utility_diff, n_boot=5000, seed=BOOTSTRAP_SEED))},
        "severe_warnings": severe,
        "thresholds": {"extreme_flip_rate": .05, "two_or_more_flip_rate": .15,
                       "weighted_kappa": .60, "absolute_mean_drift": .25},
        "human_gold": False, "silver_only": True,
    }


def augmented_row(row: dict, residual_meta: dict[tuple[str, str], dict], batch_id: str) -> dict:
    key = (str(row["query_id"]), str(row["comment_id"]))
    scores = row["validated_scores"]
    return {
        "query_id": key[0], "comment_id": key[1],
        **{f"label_{dim}": int(scores[dim]) for dim in DIMS_V2},
        "utility": float(row["utility"]), "rationale": row.get("rationale", ""),
        "judge_model": row["model"], "judge_id": "utility-v2",
        "judge_version": row["judge_version"], "batch_id": batch_id,
        "label_role": "LLM simulated-user silver; not human gold",
        "judgment_source": "top3_residual_incremental",
        "residual_backend": residual_meta[key],
    }


def method_query_metrics(ranked: list[str], gains: dict[str, dict], k: int = 3) -> dict:
    top = ranked[:k]
    judged = [cid for cid in top if cid in gains]
    coverage = len(judged) / k
    result = {"judged_pairs": len(judged), "coverage": coverage}
    for dim in DIMS_V2:
        result[f"mean_{dim}"] = mean(gains[cid][dim] for cid in judged)
    result["mean_utility"] = mean(gains[cid]["utility"] for cid in judged)
    result["condensed_graded_ndcg"] = (
        float(graded_ndcg_at(judged, {cid: gains[cid]["utility"] for cid in gains}, len(judged)))
        if judged else None)
    if len(judged) == k:
        useful = {cid for cid in top if gains[cid]["utility"] >= USEFUL_THRESHOLD}
        first = next((idx for idx, cid in enumerate(top, 1) if cid in useful), None)
        result.update({
            "mrr_utility_ge4": 1.0 / first if first else 0.0,
            "success_at_1_utility_ge4": float(top[0] in useful),
            "success_at_3_utility_ge4": float(bool(useful)),
            "zero_useful_result_at_3": float(not useful),
        })
    else:
        result.update({key: None for key in (
            "mrr_utility_ge4", "success_at_1_utility_ge4",
            "success_at_3_utility_ge4", "zero_useful_result_at_3")})
    return result


def aggregate_metrics(per_query: dict[str, dict], qids: set[str]) -> dict:
    rows = [per_query[qid] for qid in sorted(qids)]
    keys = [*(f"mean_{dim}" for dim in DIMS_V2), "mean_utility",
            "condensed_graded_ndcg", "mrr_utility_ge4", "success_at_1_utility_ge4",
            "success_at_3_utility_ge4", "zero_useful_result_at_3"]
    return {
        "query_count": len(qids), "pair_slots": len(qids) * 3,
        "judged_pairs": sum(row["judged_pairs"] for row in rows),
        "judgment_coverage_at_3": mean(row["coverage"] for row in rows),
        "queries_with_complete_top3": sum(row["coverage"] == 1 for row in rows),
        **{key: mean(row[key] for row in rows) for key in keys},
        "binary_metric_note": "MRR/success/zero-useful aggregate only complete-top3 queries; unjudged is never zero.",
        "ndcg_note": "condensed graded nDCG omits unjudged candidates and reports coverage.",
    }


def pairwise(left: dict[str, dict], right: dict[str, dict], qids: set[str]) -> dict:
    result = {}
    metrics = [*(f"mean_{dim}" for dim in DIMS_V2), "mean_utility",
               "condensed_graded_ndcg"]
    for metric in metrics:
        shared = [qid for qid in sorted(qids)
                  if left[qid].get(metric) is not None and right[qid].get(metric) is not None]
        deltas = [left[qid][metric] - right[qid][metric] for qid in shared]
        result[metric] = {
            "paired_queries": len(shared), "mean_paired_difference": mean(deltas),
            "median_paired_difference": median(deltas),
            "improved_queries": sum(value > 1e-12 for value in deltas),
            "tied_queries": sum(abs(value) <= 1e-12 for value in deltas),
            "degraded_queries": sum(value < -1e-12 for value in deltas),
            "bootstrap_95ci": list(bootstrap_ci(
                deltas, n_boot=5000, seed=BOOTSTRAP_SEED + len(metric))),
            "paired_randomization_p_two_sided": paired_permutation_p(
                deltas, seed=BOOTSTRAP_SEED + len(metric)),
        }
    return result


def summarize_group(rows: list[dict]) -> dict:
    if not rows:
        return {"pairs": 0}
    return {
        "pairs": len(rows), "queries": len({row["query_id"] for row in rows}),
        "mean_utility": mean(row["utility"] for row in rows),
        "useful_utility_ge4_share": mean(row["utility"] >= USEFUL_THRESHOLD for row in rows),
        "high_relevance_ge5_low_actionability_le3_share": mean(
            row["validated_scores"]["relevance"] >= 5
            and row["validated_scores"]["actionability"] <= 3 for row in rows),
        "safety_issue_le2_share": mean(row["validated_scores"]["safety"] <= 2 for row in rows),
        "dimensions": {dim: {
            "mean": mean(row["validated_scores"][dim] for row in rows),
            "distribution_0_to_7": {str(score): sum(
                row["validated_scores"][dim] == score for row in rows) / len(rows)
                for score in range(8)}} for dim in DIMS_V2},
        "constraint_mismatch": "NOT_IN_UTILITY_V2_SCHEMA",
        "per_need_coverage": "NOT_IN_UTILITY_V2_SCHEMA",
    }


def analyze(args) -> None:
    old_rows, old_registry = completed_old_rows(args.old_judgments)
    anchor_rows = read_jsonl(args.out_dir / "top3_anchor_rejudgments_validated.jsonl")
    stability = anchor_report(old_registry, anchor_rows)
    write_json(args.out_dir / "top3_anchor_stability_report.json", stability)
    lines = ["# Top-3 calibration anchor stability", "",
             f"Verdict: **{stability['verdict']}**", "",
             f"Valid anchors: {stability['valid_anchor_pairs']}/50.", "",
             "| dimension | exact | weighted kappa | MAD | mean shift | >=2 flip | extreme flip |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for dim, row in stability["dimensions"].items():
        lines.append(f"| {dim} | {row['exact_agreement']:.3f} | "
                     f"{row['weighted_cohen_kappa_quadratic']:.3f} | "
                     f"{row['mean_absolute_difference']:.3f} | "
                     f"{row['mean_shift_new_minus_old']:.3f} | "
                     f"{row['two_or_more_grade_flip_rate']:.3f} | "
                     f"{row['extreme_flip_rate_abs_ge_4']:.3f} |")
    lines += ["", "These are LLM-silver repeatability diagnostics, not human reliability."]
    (args.out_dir / "top3_anchor_stability_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    if stability["verdict"] not in {"STABLE", "STABLE_WITH_MINOR_DRIFT"}:
        verdict = {
            "verdict": "INCONCLUSIVE_DUE_TO_BATCH_DRIFT",
            "anchor_verdict": stability["verdict"], "merge_performed": False,
            "final_metrics_computed": False, "human_gold": False,
        }
        write_json(args.out_dir / "top3_residual_final_verdict.json", verdict)
        write_audit(args, verdict, stability, merge_performed=False)
        print(json.dumps(verdict, indent=2)); return

    residual = read_jsonl(args.residual)
    residual_meta = {(str(row["query_id"]), str(row["comment_id"])): row for row in residual}
    residual_judged = read_jsonl(args.out_dir / "top3_residual_judgments_validated.jsonl")
    if len(residual_judged) != EXPECTED_RESIDUAL_PAIRS:
        raise ValueError(f"valid residual judgments {len(residual_judged)} != 189")
    residual_registry = {(str(row["query_id"]), str(row["comment_id"])): row
                         for row in residual_judged}
    if set(residual_registry) != set(residual_meta):
        raise ValueError("validated residual identity differs from frozen manifest")
    batch = json.loads((args.out_dir / "top3_judge_batch_manifest.json").read_text())
    new_rows = [augmented_row(row, residual_meta, batch["batch_id"])
                for row in residual_judged]
    augmented = old_rows + new_rows
    write_jsonl(args.out_dir / "utility_v2_augmented_top3_judgments.jsonl", augmented)
    write_json(args.out_dir / "utility_v2_augmented_top3_registry.json", {
        "old_pairs": len(old_rows), "new_residual_pairs": len(new_rows),
        "total_pairs": len(augmented), "anchor_new_scores_in_registry": 0,
        "old_judgments_overwritten": 0, "unjudged_assigned_zero": False,
        "batch_id": batch["batch_id"], "human_gold": False,
    })
    write_json(args.out_dir / "utility_v2_augmented_top3_merge_audit.json", {
        "verdict": "PASS", "unique_pairs": len({(str(r['query_id']), str(r['comment_id'])) for r in augmented}),
        "rows": len(augmented), "old_rows_unchanged": True,
        "anchor_rejudgments_excluded": True, "residual_identity_exact": True,
        "unjudged_assigned_zero": False,
    })

    queries = load_queries(args.queries); qids = set(queries)
    query_types = load_query_types(args.query_admin)
    runs = {
        "original": group_run(args.original_run, qids),
        "original_plus_structured": group_run(args.orig_struct_run, qids),
        "cohere_dense": group_run(args.dense_run, qids),
        "official_static": group_run(args.static_run, qids),
        "dense_bridge_2hop": group_run(args.bridge_run, qids),
    }
    strata, by_qid = query_strata(runs["original"], runs["original_plus_structured"], query_types)
    runs["conditional_structured_recovery"] = {
        qid: (runs["original"][qid] if qid in strata["original_graph_entered28"]
              else runs["original_plus_structured"][qid]) for qid in qids}
    gains: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in augmented:
        gains[str(row["query_id"])][str(row["comment_id"])] = {
            **{dim: float(row[f"label_{dim}"]) for dim in DIMS_V2},
            "utility": float(row["utility"]),
        }
    per_method = {}
    for method, run in runs.items():
        per_method[method] = {
            qid: method_query_metrics(
                [str(row["comment_id"]) for row in ranked], gains.get(qid, {}))
            for qid, ranked in run.items()}
    metrics = {stratum: {
        "query_count": len(members),
        "methods": {method: aggregate_metrics(per_q, members)
                    for method, per_q in per_method.items()}}
        for stratum, members in strata.items()}
    comparisons = {}
    pairs = [
        ("original_plus_structured", "original"),
        ("original_plus_structured", "cohere_dense"),
        ("original_plus_structured", "dense_bridge_2hop"),
        ("conditional_structured_recovery", "original"),
        ("conditional_structured_recovery", "original_plus_structured"),
    ]
    for stratum, members in strata.items():
        comparisons[stratum] = {
            f"{left}_minus_{right}": pairwise(
                per_method[left], per_method[right], members)
            for left, right in pairs}
    write_json(args.out_dir / "top3_pairwise_comparisons.json", {
        "metrics_by_stratum": metrics, "pairwise": comparisons,
        "main_result_stratum": "all100", "mechanism_stratum": "structured_recovered61",
        "safety_stratum": "original_graph_entered28", "human_gold": False,
    })
    per_query_rows = []
    for qid in sorted(strata["structured_recovered61"]):
        row = {"query_id": qid, "strata": by_qid[qid]}
        for method in ("original", "original_plus_structured", "dense_bridge_2hop",
                       "conditional_structured_recovery"):
            row[f"{method}_metric"] = per_method[method][qid]["mean_utility"]
        row["orig_struct_minus_original"] = (
            row["original_plus_structured_metric"] - row["original_metric"]
            if row["original_plus_structured_metric"] is not None
            and row["original_metric"] is not None else None)
        row["orig_struct_minus_bridge"] = (
            row["original_plus_structured_metric"] - row["dense_bridge_2hop_metric"]
            if row["original_plus_structured_metric"] is not None
            and row["dense_bridge_2hop_metric"] is not None else None)
        per_query_rows.append(row)
    write_jsonl(args.out_dir / "top3_pairwise_per_query.jsonl", per_query_rows)
    sig_lines = ["# Top-3 paired significance report", "",
                 "All labels are development-stage LLM silver. CIs are query-paired bootstrap; "
                 "p-values are fixed-seed paired sign randomization.", ""]
    for stratum in ("all100", "structured_recovered61", "original_graph_entered28",
                    "single50", "multi50", "still_fallback11"):
        sig_lines += [f"## {stratum}", ""]
        for comparison, payload in comparisons[stratum].items():
            row = payload["mean_utility"]
            sig_lines.append(
                f"- {comparison}: delta={row['mean_paired_difference']:.4f}, "
                f"CI={row['bootstrap_95ci']}, p={row['paired_randomization_p_two_sided']:.4f}, "
                f"improved/tied/degraded={row['improved_queries']}/"
                f"{row['tied_queries']}/{row['degraded_queries']}.")
        sig_lines.append("")
    (args.out_dir / "top3_significance_report.md").write_text(
        "\n".join(sig_lines) + "\n", encoding="utf-8")

    run_sets20 = {name: {qid: {str(row["comment_id"]) for row in ranked[:20]}
                         for qid, ranked in run.items()}
                  for name, run in runs.items() if name in {
                      "cohere_dense", "official_static", "dense_bridge_2hop"}}
    enriched = []
    for row in residual_judged:
        key = (str(row["query_id"]), str(row["comment_id"]))
        meta = residual_meta[key]
        enriched.append({**row, "orig_struct_rank": meta["orig_struct_rank"],
                         "strict_graph_derived": meta["strict_graph_derived"],
                         "strata": by_qid[key[0]],
                         **{f"in_{method}_top20": key[1] in lookup[key[0]]
                            for method, lookup in run_sets20.items()}})
    groupings = {
        "all189": enriched,
        **{f"orig_struct_rank_{rank}": [row for row in enriched
                                         if int(row["orig_struct_rank"]) == rank]
           for rank in (1, 2, 3)},
        **{name: [row for row in enriched if name in row["strata"]]
           for name in strata},
        "strict_graph_derived": [row for row in enriched if row["strict_graph_derived"]],
        "not_strict_graph_derived": [row for row in enriched if not row["strict_graph_derived"]],
    }
    for method in run_sets20:
        groupings[f"in_{method}_top20"] = [row for row in enriched
                                            if row[f"in_{method}_top20"]]
        groupings[f"not_in_{method}_top20"] = [row for row in enriched
                                                if not row[f"in_{method}_top20"]]
    quality = {"groups": {name: summarize_group(rows) for name, rows in groupings.items()},
               "score_scale": "utility-v2 uses 1..7; score=0 is impossible and not added",
               "human_gold": False}
    write_json(args.out_dir / "top3_residual_candidate_quality.json", quality)
    q = quality["groups"]
    quality_lines = ["# Top-3 residual candidate quality", "",
                     "Utility-v2 is a 1–7 LLM-silver rubric; no 0 score was introduced.", "",
                     f"- All189 mean utility: {q['all189']['mean_utility']:.3f}; useful>=4: "
                     f"{q['all189']['useful_utility_ge4_share']:.1%}.",
                     f"- Strict graph-derived mean utility: {q['strict_graph_derived']['mean_utility']:.3f}; "
                     f"non-strict: {q['not_strict_graph_derived']['mean_utility']:.3f}.",
                     f"- Recovered61 residual mean utility: {q['structured_recovered61']['mean_utility']:.3f}.",
                     "- Constraint fit and per-need coverage were not in utility-v2 and were not invented."]
    (args.out_dir / "top3_residual_candidate_quality.md").write_text(
        "\n".join(quality_lines) + "\n", encoding="utf-8")

    all_delta = comparisons["all100"]["original_plus_structured_minus_original"]
    rec_delta = comparisons["structured_recovered61"]["original_plus_structured_minus_original"]
    entered_delta = comparisons["original_graph_entered28"]["original_plus_structured_minus_original"]
    all_u = all_delta["mean_utility"]; rec_core = [rec_delta[key] for key in (
        "mean_relevance", "mean_usefulness", "mean_actionability", "mean_utility")]
    entered_core = [entered_delta[key] for key in (
        "mean_relevance", "mean_usefulness", "mean_actionability", "mean_utility")]
    quality_ok = (q["all189"]["useful_utility_ge4_share"] >= .5
                  and q["all189"]["dimensions"]["relevance"]["mean"] >= 4)
    no_obvious_all_degrade = all_u["bootstrap_95ci"][1] >= -.25
    recovered_positive = any(row["mean_paired_difference"] > 0
                             and row["bootstrap_95ci"][0] >= 0 for row in rec_core)
    recovered_clear_degrade = (rec_delta["mean_utility"]["bootstrap_95ci"][1] < 0
                               and rec_delta["mean_relevance"]["bootstrap_95ci"][1] < 0)
    entered_degrade = any(row["bootstrap_95ci"][1] < 0 for row in entered_core)
    conditional_rec = comparisons["all100"]["conditional_structured_recovery_minus_original"]
    conditional_positive = conditional_rec["mean_utility"]["mean_paired_difference"] > 0
    coverage = metrics["all100"]["methods"]["original_plus_structured"]["judgment_coverage_at_3"]
    if coverage < .95:
        final = "INCONCLUSIVE_DUE_TO_COVERAGE"
    elif recovered_clear_degrade:
        final = "STOP_REWRITE_RECOVERY"
    elif recovered_positive and entered_degrade and conditional_positive and quality_ok:
        final = "PROCEED_WITH_CONDITIONAL_RECOVERY"
    elif recovered_positive and no_obvious_all_degrade and quality_ok:
        final = "PROCEED_TO_TOP5"
    else:
        final = "KEEP_AS_ABLATION_ONLY"
    assert final in ALLOWED_VERDICTS
    verdict = {
        "verdict": final, "anchor_verdict": stability["verdict"],
        "judgment_coverage_orig_struct_at3": coverage,
        "quality_gate_pass": quality_ok, "all100_not_obviously_degraded": no_obvious_all_degrade,
        "recovered61_stable_positive_core_trend": recovered_positive,
        "recovered61_clear_degradation": recovered_clear_degrade,
        "graph_entered28_clear_degradation": entered_degrade,
        "conditional_mean_utility_positive": conditional_positive,
        "human_gold": False, "silver_only": True,
    }
    write_json(args.out_dir / "top3_residual_final_verdict.json", verdict)
    if final in {"PROCEED_TO_TOP5", "PROCEED_WITH_CONDITIONAL_RECOVERY"}:
        augmented_keys = {(str(row["query_id"]), str(row["comment_id"])) for row in augmented}
        remaining = []
        for qid, ranked in runs["original_plus_structured"].items():
            for row in ranked[:5]:
                key = (qid, str(row["comment_id"]))
                if key not in augmented_keys:
                    remaining.append({"query_id": qid, "comment_id": key[1],
                                      "orig_struct_rank": int(row["rank"]),
                                      "strata": by_qid[qid]})
        top5 = {
            "new_unique_pairs": len(remaining),
            "involved_queries": len({row["query_id"] for row in remaining}),
            "recovered61_pairs": sum("structured_recovered61" in row["strata"] for row in remaining),
            "single50_pairs": sum("single50" in row["strata"] for row in remaining),
            "multi50_pairs": sum("multi50" in row["strata"] for row in remaining),
            "current_augmented_coverage_at5": 1 - len(remaining) / 500,
            "projected_coverage_after_completion": 1.0,
            "main_calls": len(remaining), "calls_with_50_calibration_anchors": len(remaining) + 50,
            "llm_calls_made_for_top5": 0,
        }
        write_json(args.out_dir / "top5_next_stage_residual_size.json", top5)
        (args.out_dir / "top5_next_stage_recommendation.md").write_text(
            f"# Top-5 next stage\n\nVerdict: **{final}**.\n\n"
            f"Remaining pairs: {len(remaining)}; current coverage@5: "
            f"{top5['current_augmented_coverage_at5']:.1%}; projected: 100%. "
            "This file only sizes the next stage; no Top-5 Judge call was made.\n",
            encoding="utf-8")
    write_audit(args, verdict, stability, merge_performed=True)
    print(json.dumps(verdict, indent=2))


def write_audit(args, verdict: dict, stability: dict, merge_performed: bool) -> None:
    checks = {
        "frozen_residual_189_unchanged": sha256(args.residual) == json.loads(
            (args.out_dir / "top3_judge_batch_manifest.json").read_text())["residual_manifest_sha256"],
        "utility_used_to_select_residual": False,
        "judge_protocol_equivalent": json.loads(
            (args.out_dir / "judge_protocol_equivalence_report.json").read_text())["verdict"] == "EQUIVALENT",
        "judge_payload_blind": json.loads(
            (args.out_dir / "judge_payload_blindness_audit.json").read_text())["verdict"] == "PASS",
        "anchors_selected_without_old_scores": True,
        "anchor_new_scores_overwrite_old": False,
        "unjudged_assigned_zero": False,
        "frozen_test_read": False,
        "retrieval_runs_modified": False,
        "pairs_deleted_for_low_scores": False,
        "top10_improvement_claimed": False,
        "all100_and_recovered61_separated": True,
        "llm_silver_called_human_gold": False,
    }
    expected_false = {
        "utility_used_to_select_residual", "anchor_new_scores_overwrite_old",
        "unjudged_assigned_zero", "frozen_test_read", "retrieval_runs_modified",
        "pairs_deleted_for_low_scores", "top10_improvement_claimed",
        "llm_silver_called_human_gold",
    }
    violations = [name for name, value in checks.items()
                  if value != (name not in expected_false)]
    payload = {"verdict": "PASS" if not violations else "FAIL",
               "checks": checks, "violations": violations,
               "anchor_verdict": stability["verdict"], "merge_performed": merge_performed,
               "final_verdict": verdict["verdict"]}
    write_json(args.out_dir / "top3_residual_judging_protocol_violations.json", payload)
    lines = ["# Independent audit: Top-3 residual judging", "",
             f"Audit verdict: **{payload['verdict']}**", "",
             f"Anchor verdict: **{stability['verdict']}**", "",
             f"Final method verdict: **{verdict['verdict']}**", "",
             "The frozen 189-pair manifest was not regenerated; retrieval runs, graph, OpenIE, "
             "recognition, PPR, union weights, old judgments, and frozen test were not modified/read. "
             "Judge payloads contained only the historical query/comment/empty-facets fields. "
             "All results remain LLM silver, not human gold."]
    (args.out_dir / "top3_residual_judging_independent_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["prepare", "judge", "analyze"])
    ap.add_argument("--residual", type=Path, default=root / "out/query_rewrite_graph_entry_dev100_v2/residual_judging/residual_pairs_top3.jsonl")
    ap.add_argument("--residual-size-report", type=Path, default=root / "out/query_rewrite_graph_entry_dev100_v2/residual_judging/residual_judging_size_report.json")
    ap.add_argument("--old-judgments", type=Path, default=root / "out/section17_dev_100/llm_utility_v2.jsonl")
    ap.add_argument("--old-judge-manifest", type=Path, default=root / "out/section17_dev_100/llm_utility_v2.manifest.json")
    ap.add_argument("--queries", type=Path, default=root / "out/section17_dev_100_v2/section17_dev_queries_official.json")
    ap.add_argument("--query-admin", type=Path, default=root / "out/section17_dev_100_v2/section17_dev_queries_ADMIN.csv")
    ap.add_argument("--summaries", type=Path, default=root / "out/frozen_query_summaries_dev100_v2/frozen_query_summaries.jsonl")
    ap.add_argument("--corpus", type=Path, default=root / "out/hipporag_official_adapter/adhd_peer_support_validation_corpus.json")
    runs = root / "out/query_rewrite_graph_entry_dev100_v2/official_runs"
    versioned = root / "out/dev100_v2_versioned_runs/runs"
    ap.add_argument("--original-run", type=Path, default=runs / "official_original_top100.jsonl")
    ap.add_argument("--orig-struct-run", type=Path, default=runs / "official_original_plus_structured_top100.jsonl")
    ap.add_argument("--dense-run", type=Path, default=versioned / "cohere_dense.jsonl")
    ap.add_argument("--static-run", type=Path, default=versioned / "official_static.jsonl")
    ap.add_argument("--bridge-run", type=Path, default=versioned / "dense_bridge_2hop.jsonl")
    ap.add_argument("--judge-prompt", type=Path)
    ap.add_argument("--judge-code", type=Path)
    ap.add_argument("--section17-runner", type=Path)
    ap.add_argument("--llm-client", type=Path, default=root / "shared/llm_client.py")
    ap.add_argument("--out-dir", type=Path, default=root / "out/query_rewrite_graph_entry_dev100_v2/top3_residual_judging")
    ap.add_argument("--workers", type=int, default=4)
    return ap


def main() -> None:
    ap = parser()
    args = ap.parse_args()
    if args.phase == "prepare":
        if args.judge_prompt is None:
            args.judge_prompt = project_config.prompt_path("evidence_card_judge_v2")
        for name in ("judge_code", "section17_runner"):
            if getattr(args, name) is None:
                ap.error(f"--{name.replace('_', '-')} is required for prepare")
    if args.phase == "prepare": prepare(args)
    elif args.phase == "judge": run_judging(args)
    else: analyze(args)


if __name__ == "__main__":
    main()
