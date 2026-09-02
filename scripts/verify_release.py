#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOTS = (
    "data_preparation",
    "candidate_pool",
    "fusion",
    "utility_scoring",
    "evidence_selection",
    "evaluation",
    "figures",
    "configuration",
    "shared",
)
PROJECT_TITLE = (
    "Utility-Aware Cross-Thread Evidence Retrieval and Selection for RAG in "
    "Online ADHD Communities"
)

REQUIRED_DIRECTORIES = {
    "data_preparation/sampling",
    "data_preparation/entity_processing",
    "candidate_pool/retrieval",
    "candidate_pool/graph_construction",
    "fusion",
    "utility_scoring/learned_diffusion",
    "utility_scoring/annotation",
    "evidence_selection",
    "evaluation",
    "figures",
    "configuration",
    "shared",
    "models/primary",
    "models/supplementary",
}
OBSOLETE_DIRECTORIES = {"graph_rag", "models/sensitivity"}
DOCUMENTED_ENTRYPOINTS = {
    "data_preparation/sampling/freeze_research_data_partitions.py",
    "candidate_pool/run_official_hipporag_bedrock.py",
    "evaluation/run_evidence_signal_triangulation.py",
    "fusion/analyze_rq2a_graph_budget_sweep.py",
    "fusion/run_depth_graph_utility_community_frontier.py",
    "utility_scoring/build_stage2_redesign_features.py",
    "utility_scoring/fit_lambdamart_transfer.py",
    "utility_scoring/run_lightweight_scorer_search_dev300.py",
    "utility_scoring/run_stage2_redesign_crossencoder.py",
    "evidence_selection/run_selection_action_space_repair.py",
    "evaluation/confirmatory_test200_rq2b.py",
}

FORBIDDEN_ANYWHERE = {"__pycache__", "prompts"}
FORBIDDEN_TOP_LEVEL = {"out", "data", "webapp"}
FORBIDDEN_SUFFIXES = {".pyc", ".jsonl", ".csv", ".parquet", ".zst", ".pt", ".bin"}
ALLOWED_MODEL_SUFFIXES = {".safetensors", ".cbm", ".joblib"}
PATTERNS = {
    "AWS key": re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    "Requesty key": re.compile(r"rqsty-sk-[A-Za-z0-9_-]{10,}"),
    "OpenRouter key": re.compile(r"sk-or-v1-[A-Za-z0-9]{10,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "absolute user path": re.compile(r"/(?:Users|home)/[^/\s]+/"),
    "internal process name": re.compile(r"\b(?:Codex|Cowork|Nancy)\b", re.I),
    "embedded instruction": re.compile(r"\bYou are (?:an?|the)\b", re.I),
    "embedded scoring prompt": re.compile(r"Rate the (?:candidate|answer|comment)", re.I),
}


def module_exists(module: str) -> bool:
    relative = ROOT.joinpath(*module.split("."))
    return relative.with_suffix(".py").is_file() or (relative / "__init__.py").is_file()


errors: list[str] = []
for relative in sorted(REQUIRED_DIRECTORIES):
    if not (ROOT / relative).is_dir():
        errors.append(f"missing documented directory: {relative}")
for relative in sorted(OBSOLETE_DIRECTORIES):
    if (ROOT / relative).exists():
        errors.append(f"obsolete directory retained: {relative}")
for relative in sorted(DOCUMENTED_ENTRYPOINTS):
    if not (ROOT / relative).is_file():
        errors.append(f"missing documented entry point: {relative}")

for path in ROOT.rglob("*"):
    relative = path.relative_to(ROOT)
    if ".git" in relative.parts:
        continue
    if path.is_symlink():
        errors.append(f"symlink: {relative}")
        continue
    if not path.is_file():
        continue
    if any(part in FORBIDDEN_ANYWHERE for part in relative.parts):
        errors.append(f"forbidden path: {relative}")
    if relative.parts[0] in FORBIDDEN_TOP_LEVEL:
        errors.append(f"forbidden path: {relative}")
    if path.suffix in FORBIDDEN_SUFFIXES:
        errors.append(f"forbidden file type: {relative}")
    if path.suffix in ALLOWED_MODEL_SUFFIXES and relative.parts[0] != "models":
        errors.append(f"model artefact outside models/: {relative}")
    if path.stat().st_size > 500_000_000:
        errors.append(f"oversize file: {relative}")
    is_binary_model = path.suffix in ALLOWED_MODEL_SUFFIXES or (
        relative.parts[0] == "models" and path.name == "tokenizer.json"
    )
    text = "" if is_binary_model else path.read_text(encoding="utf-8", errors="ignore")
    if not is_binary_model and relative != Path("scripts/verify_release.py"):
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label}: {relative}")

expected_models = {
    "models/primary/catboost_yetirank/model.cbm",
    "models/primary/catboost_yetirank/scaler.joblib",
    "models/primary/utility_crossencoder_256/model.safetensors",
    "models/supplementary/utility_crossencoder_512/model.safetensors",
}
for relative in expected_models:
    if not (ROOT / relative).is_file():
        errors.append(f"missing reported model: {relative}")

python_files = sorted(
    path
    for package in PACKAGE_ROOTS
    for path in (ROOT / package).rglob("*.py")
)
for path in python_files:
    try:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        tree = ast.parse(source)
    except Exception as exc:
        errors.append(f"python parse: {path.relative_to(ROOT)}: {exc}")
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        else:
            continue
        for module in modules:
            if module.split(".", 1)[0] in PACKAGE_ROOTS and not module_exists(module):
                errors.append(
                    f"missing internal import {module}: {path.relative_to(ROOT)}"
                )

manifest_path = ROOT / "MANIFEST.json"
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception as exc:
    errors.append(f"manifest parse: {exc}")
else:
    if manifest.get("project") != PROJECT_TITLE:
        errors.append("manifest project title does not match the release")
    rows = list(manifest.get("files") or [])
    manifest_files = {str(row.get("path")): row for row in rows}
    if len(manifest_files) != len(rows):
        errors.append("manifest contains duplicate file paths")
    actual_files = {
        path.relative_to(ROOT).as_posix(): path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(ROOT).parts
        and path != manifest_path
    }
    for relative in sorted(set(actual_files) - set(manifest_files)):
        errors.append(f"manifest missing file: {relative}")
    for relative in sorted(set(manifest_files) - set(actual_files)):
        errors.append(f"manifest has stale file: {relative}")
    for relative in sorted(set(actual_files) & set(manifest_files)):
        path = actual_files[relative]
        row = manifest_files[relative]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if int(row.get("bytes", -1)) != path.stat().st_size:
            errors.append(f"manifest byte count mismatch: {relative}")
        if row.get("sha256") != digest:
            errors.append(f"manifest hash mismatch: {relative}")

if errors:
    print("RELEASE CHECK FAILED")
    print("\n".join(sorted(set(errors))))
    raise SystemExit(1)

print(
    "RELEASE CHECK PASSED: "
    f"{len(python_files)} Python files; "
    f"{len(expected_models)} required model artefacts"
)
