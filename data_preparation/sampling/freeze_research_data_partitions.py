#!/usr/bin/env python3
"""Top-level, configuration-driven freeze for graph and evaluation partitions.

This tool does not invent a new sampler.  It orchestrates and audits the
existing graph-corpus sampler, historical construction split, natural-query
eligibility gate, and final nested dev/test freezer under one versioned
contract.

The hosted-model boundary is explicit: LLM annotations/extractions are frozen
by hash, because a new API call is not byte deterministic.  All deterministic
selection and graph-assembly inputs around that boundary are locked here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configuration" / "data_partitions_v1.yaml"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "out" / "research_data_partitions_v1" / "manifest.json"
)


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("version") != "adhd-graph-rag-data-partitions-v1":
        raise ValueError(f"unsupported partition config version: {raw.get('version')}")
    return raw


def read_ids(path: Path, field: str) -> tuple[int, set[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = {
        str(row.get(field) or row.get("post_id") or row.get("query_id") or "").strip()
        for row in rows
    }
    ids.discard("")
    return len(rows), ids


def id_set_sha256(ids: set[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_locked_artifact(name: str, spec: dict) -> tuple[dict, set[str] | None]:
    path = resolve(spec["path"])
    if not path.exists():
        raise FileNotFoundError(f"{name}: missing locked artifact: {path}")
    actual = {
        "path": str(path.relative_to(PROJECT_ROOT)
                    if path.is_relative_to(PROJECT_ROOT) else path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    for key in ("bytes", "sha256"):
        if key in spec and actual[key] != spec[key]:
            raise ValueError(
                f"{name}: {key} mismatch: expected={spec[key]} actual={actual[key]}"
            )
    ids = None
    if "id_field" in spec:
        rows, ids = read_ids(path, str(spec["id_field"]))
        actual.update({
            "rows": rows,
            "unique_ids": len(ids),
            "id_set_sha256": id_set_sha256(ids),
        })
        for key in ("rows", "unique_ids", "id_set_sha256"):
            if actual[key] != spec[key]:
                raise ValueError(
                    f"{name}: {key} mismatch: expected={spec[key]} "
                    f"actual={actual[key]}"
                )
    return actual, ids


def _ids_for_csv(path: str, field: str = "query_id") -> set[str]:
    return read_ids(resolve(path), field)[1]


def verify_invariants(cfg: dict, artifact_ids: dict[str, set[str] | None]) -> dict:
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, **details: Any) -> None:
        checks[name] = {"passed": passed, **details}
        if not passed:
            raise ValueError(f"partition invariant failed: {name}: {details}")

    raw_ids = artifact_ids["expanded_graph_sample"] or set()
    cap_ids = artifact_ids["expanded_graph_cap30"] or set()
    record(
        "expanded_graph_sample_ids_equal_cap30_ids",
        raw_ids == cap_ids,
        raw_n=len(raw_ids),
        cap30_n=len(cap_ids),
        symmetric_difference_n=len(raw_ids ^ cap_ids),
    )

    construction = {
        name: artifact_ids[name] or set()
        for name in (
            "construction_validation600",
            "construction_test200",
            "construction_reserve1142",
        )
    }
    construction_overlaps = {
        "validation_vs_test": len(
            construction["construction_validation600"]
            & construction["construction_test200"]
        ),
        "validation_vs_reserve": len(
            construction["construction_validation600"]
            & construction["construction_reserve1142"]
        ),
        "test_vs_reserve": len(
            construction["construction_test200"]
            & construction["construction_reserve1142"]
        ),
    }
    record(
        "construction_validation_test_reserve_pairwise_disjoint",
        not any(construction_overlaps.values()),
        overlaps=construction_overlaps,
    )

    eligible = artifact_ids["eligible_unseen1137"] or set()
    extraction = artifact_ids["graph_extraction_input"] or set()
    expected_eligible = construction["construction_reserve1142"] - extraction
    record(
        "eligible_unseen_equals_reserve_minus_graph_extraction_posts",
        eligible == expected_eligible,
        reserve_n=len(construction["construction_reserve1142"]),
        excluded_same_post_n=len(
            construction["construction_reserve1142"] & extraction
        ),
        expected_n=len(expected_eligible),
        actual_n=len(eligible),
        symmetric_difference_n=len(eligible ^ expected_eligible),
    )

    closed = artifact_ids["closed_structural600"] or set()
    record(
        "closed_structural600_matches_construction_validation600",
        closed == construction["construction_validation600"],
        closed_n=len(closed),
        construction_n=len(construction["construction_validation600"]),
        symmetric_difference_n=len(
            closed ^ construction["construction_validation600"]
        ),
    )

    utility_pool = artifact_ids["utility_pool173_candidates"] or set()
    partial_manifest = json.loads(
        resolve(
            cfg["paper_evaluation_cohorts"]["utility_pool173"][
                "partial_snapshot_manifest"
            ]
        ).read_text(encoding="utf-8")
    )
    shortcut_manifest = json.loads(
        resolve(
            cfg["paper_evaluation_cohorts"]["utility_pool173"][
                "result_manifest"
            ]
        ).read_text(encoding="utf-8")
    )
    utility_spec = cfg["paper_evaluation_cohorts"]["utility_pool173"]
    lopo_counts = {
        int(
            row["cross_post_llm_utility"][
                "queries_with_cross_post_judgments"
            ]
        )
        for row in shortcut_manifest["systems"].values()
    }
    record(
        "utility_pool173_is_subset_of_closed_structural600",
        (
            utility_pool <= closed
            and len(utility_pool) == int(utility_spec["query_count"])
            and int(partial_manifest[
                "queries_with_at_least_one_retained_structural_gold"
            ]) == int(utility_spec["query_count"])
            and lopo_counts == {
                int(utility_spec["lopo_evaluable_query_count"])
            }
        ),
        utility_pool_n=len(utility_pool),
        outside_closed_n=len(utility_pool - closed),
        partial_snapshot_query_n=partial_manifest[
            "queries_with_at_least_one_retained_structural_gold"
        ],
        lopo_evaluable_counts=sorted(lopo_counts),
        selection_is_random=False,
    )

    pass_labels = artifact_ids["query_structure_pass_labels"] or set()
    confirmation = artifact_ids["single_multi_confirmation600"] or set()
    confirmation_labels = (
        artifact_ids["single_multi_confirmation_labels"] or set()
    )
    library = artifact_ids["single_multi_library400"] or set()
    record(
        "single_multi_query_structure_lineage_is_nested_in_eligible1137",
        (
            pass_labels <= eligible
            and confirmation <= pass_labels
            and confirmation_labels == confirmation
            and library <= confirmation
        ),
        eligible_n=len(eligible),
        pass1_or_pass2_labeled_query_n=len(pass_labels),
        confirmation_n=len(confirmation),
        confirmation_label_query_n=len(confirmation_labels),
        balanced_library_n=len(library),
        pass_labels_outside_eligible_n=len(pass_labels - eligible),
        confirmation_outside_labeled_n=len(confirmation - pass_labels),
        library_outside_confirmation_n=len(library - confirmation),
    )

    development = artifact_ids["development398"] or set()
    record(
        "development398_is_subset_of_single_multi_library400",
        development <= library,
        development_n=len(development),
        missing_n=len(development - library),
    )

    final_test = artifact_ids["cross_thread_test200"] or set()
    record(
        "development398_and_cross_thread_test200_disjoint",
        not (development & final_test),
        overlap_n=len(development & final_test),
    )
    record(
        "cross_thread_test200_disjoint_from_construction_test200",
        not (final_test & construction["construction_test200"]),
        overlap_n=len(final_test & construction["construction_test200"]),
    )
    record(
        "cross_thread_test200_disjoint_from_graph_extraction_input",
        not (final_test & extraction),
        overlap_n=len(final_test & extraction),
    )

    exclusion_ids: set[str] = set()
    for path in cfg["evaluation_freeze"]["test_exclusion_files"]:
        exclusion_ids |= _ids_for_csv(path)
    record(
        "cross_thread_test200_disjoint_from_all_development_exclusion_streams",
        not (final_test & exclusion_ids),
        exclusion_union_n=len(exclusion_ids),
        overlap_n=len(final_test & exclusion_ids),
    )
    reranker_split_path = resolve(
        cfg["evaluation_freeze"]["reranker_cv"]["output"]
    )
    reranker_split = json.loads(
        reranker_split_path.read_text(encoding="utf-8")
    )
    fold_violations = []
    for row in reranker_split["rows"]:
        train = set(row["train_query_ids"])
        valid = set(row["validation_query_ids"])
        if train & valid or train | valid != development:
            fold_violations.append(
                f"repeat={row['repeat']} fold={row['fold']}"
            )
    record(
        "reranker_grouped_cv_covers_development398_only",
        (
            reranker_split["audit"]["verdict"] == "PASS"
            and reranker_split["audit"]["unique_queries"] == len(development)
            and not fold_violations
        ),
        fold_rows=len(reranker_split["rows"]),
        unique_queries=reranker_split["audit"]["unique_queries"],
        violations=fold_violations,
        outcome_or_retrieval_features_used=reranker_split[
            "outcome_or_retrieval_features_used"
        ],
    )
    return checks


def verify(cfg: dict, config_path: Path = DEFAULT_CONFIG) -> dict:
    artifacts: dict[str, dict] = {}
    artifact_ids: dict[str, set[str] | None] = {}
    for name, spec in cfg["locked_artifacts"].items():
        artifacts[name], artifact_ids[name] = verify_locked_artifact(name, spec)
    checks = verify_invariants(cfg, artifact_ids)
    code = {
        path: sha256(resolve(path))
        for path in cfg.get("code_entrypoints", [])
    }
    environments = {
        path: sha256(resolve(path))
        for path in cfg.get("environment_lockfiles", [])
    }
    return {
        "version": cfg["version"],
        "protocol_frozen_at": cfg["protocol_frozen_at"],
        "config_sha256": sha256(config_path),
        "status": "passed",
        "role_boundaries": {
            "graph_sampling_frame": "expanded_graph_sample",
            "graph_construction_cohorts": [
                "construction_validation600",
                "construction_test200",
            ],
            "development": "development398",
            "confirmatory_cross_thread_test": "cross_thread_test200",
            "paper_evaluation_cohorts": {
                "closed_structural": "closed_structural600",
                "historical_utility_pool": "utility_pool173_candidates",
                "historical_lopo_evaluable_queries": 170,
                "balanced_query_structure_library": "single_multi_library400",
                "development": "development398",
                "confirmatory_test": "cross_thread_test200",
            },
            "hosted_model_outputs": "frozen_by_hash_not_recalled_for_reproduction",
        },
        "code_sha256": code,
        "environment_sha256": environments,
        "artifacts": artifacts,
        "invariants": checks,
        "test_content_exposed": False,
        "retrieval_or_judgments_run": False,
    }


def freeze_eval_command(
    cfg: dict,
    out_dir: Path,
    *,
    single_multi_library: Path | None = None,
    test_exclusion_files: list[Path] | None = None,
    current_dev: Path | None = None,
    eligible_unseen: Path | None = None,
    audit_splits: list[Path] | None = None,
    exposure_audit_files: list[Path] | None = None,
) -> list[str]:
    spec = cfg["evaluation_freeze"]
    command = [
        sys.executable,
        str(resolve(spec["script"])),
        "--single-multi-library",
        str(single_multi_library or resolve(spec["single_multi_library"])),
        "--current-dev",
        str(current_dev or resolve(spec["current_development"])),
        "--eligible-unseen",
        str(eligible_unseen or resolve(spec["eligible_unseen"])),
        "--out-dir", str(out_dir),
        "--dev-seed", str(spec["development_seed"]),
        "--test-seed", str(spec["test_seed"]),
        "--test-n", str(spec["test_n"]),
        "--balance", str(spec["balance"]),
        "--frozen-at", str(spec["frozen_at"]),
    ]
    for query_id in spec["development_exclude_ids"]:
        command.extend(["--development-exclude-id", str(query_id)])
    exclusions = (
        test_exclusion_files
        if test_exclusion_files is not None
        else [resolve(path) for path in spec["test_exclusion_files"]]
    )
    for path in exclusions:
        command.extend(["--test-exclude-file", str(path)])
    audits = (
        audit_splits
        if audit_splits is not None
        else [resolve(path) for path in spec["audit_splits"]]
    )
    for path in audits:
        command.extend(["--audit-split", str(path)])
    exposure_files = (
        exposure_audit_files
        if exposure_audit_files is not None
        else [resolve(path) for path in spec["exposure_audit_files"]]
    )
    for path in exposure_files:
        command.extend(["--exposure-audit-file", str(path)])
    return command


def reranker_split_command(cfg: dict, out_path: Path) -> list[str]:
    evaluation = cfg["evaluation_freeze"]
    spec = evaluation["reranker_cv"]
    command = [
        sys.executable,
        str(resolve(spec["script"])),
        "--development-admin",
        str(resolve(evaluation["canonical_output_dir"])
            / "development_queries_398_ADMIN.csv"),
        "--out",
        str(out_path),
        "--folds",
        str(spec["folds"]),
        "--seeds",
    ]
    command.extend(str(seed) for seed in spec["seeds"])
    return command


def _normalise_evaluation_manifest_provenance(value: dict) -> dict:
    """Remove absolute source locations while retaining source identities."""
    normalised = json.loads(json.dumps(value))
    sources = normalised["sources"]
    for name in ("single_multi_library", "current_dev", "eligible_unseen"):
        sources[name]["path"] = f"<{name}>"
    for name in (
        "test_exclusion_files",
        "audit_splits",
        "exposure_audit_files",
    ):
        sources[name] = sorted(
            sources[name].values(),
            key=lambda record: json.dumps(record, sort_keys=True),
        )
    # Distribution-reference audits are keyed by the path of each comparison
    # split. The path is provenance display only; the complete audit payload is
    # retained and sorted so content or parameter changes still fail replay.
    reference_audits = normalised["cross_thread_test"].get(
        "reference_distribution_audits"
    )
    if isinstance(reference_audits, dict):
        normalised["cross_thread_test"]["reference_distribution_audits"] = (
            sorted(
                reference_audits.values(),
                key=lambda record: json.dumps(record, sort_keys=True),
            )
        )
    exposure_audits = (
        normalised["cross_thread_test"]
        .get("prior_query_only_exposure", {})
        .get("audits")
    )
    if isinstance(exposure_audits, dict):
        normalised["cross_thread_test"]["prior_query_only_exposure"][
            "audits"
        ] = sorted(
            exposure_audits.values(),
            key=lambda record: json.dumps(record, sort_keys=True),
        )
    return normalised


def rebuild_evaluation(
    cfg: dict,
    out_dir: Path,
    *,
    single_multi_library: Path | None = None,
    test_exclusion_files: list[Path] | None = None,
    current_dev: Path | None = None,
    eligible_unseen: Path | None = None,
    audit_splits: list[Path] | None = None,
    exposure_audit_files: list[Path] | None = None,
    expected_outputs: dict | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    pythonpath = str(PROJECT_ROOT)
    env["PYTHONPATH"] = (
        pythonpath
        if not env.get("PYTHONPATH")
        else pythonpath + os.pathsep + env["PYTHONPATH"]
    )
    completed = subprocess.run(
        freeze_eval_command(
            cfg,
            out_dir,
            single_multi_library=single_multi_library,
            test_exclusion_files=test_exclusion_files,
            current_dev=current_dev,
            eligible_unseen=eligible_unseen,
            audit_splits=audit_splits,
            exposure_audit_files=exposure_audit_files,
        ),
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    split_command = reranker_split_command(
        cfg, out_dir / "reranker_grouped_query_split_manifest.json"
    )
    split_command[3] = str(out_dir / "development_queries_398_ADMIN.csv")
    split_completed = subprocess.run(
        split_command,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    canonical = (
        None
        if expected_outputs is not None
        else resolve(cfg["evaluation_freeze"]["canonical_output_dir"])
    )
    byte_names = [
        "development_queries_398_ADMIN.csv",
        "development_queries_398.csv",
        "development_queries_398_official.json",
        "cross_thread_test_queries_200_ADMIN.csv",
        "cross_thread_test_ids_200.csv",
        "reranker_grouped_query_split_manifest.json",
    ]
    comparison = {}
    for name in byte_names:
        rebuilt = out_dir / name
        rebuilt_hash = sha256(rebuilt)
        frozen_hash = (
            str(expected_outputs["files"][name])
            if expected_outputs is not None
            else sha256(canonical / name)
        )
        comparison[name] = {
            "rebuilt_sha256": rebuilt_hash,
            "expected_sha256": frozen_hash,
            "byte_identical": rebuilt_hash == frozen_hash,
        }
    rebuilt_manifest_path = out_dir / "manifest.json"
    rebuilt_manifest = json.loads(
        rebuilt_manifest_path.read_text(encoding="utf-8")
    )
    normalised_rebuilt = _normalise_evaluation_manifest_provenance(
        rebuilt_manifest
    )
    normalised_rebuilt_sha = _json_object_sha256(normalised_rebuilt)
    if expected_outputs is not None:
        frozen_manifest_hash = str(expected_outputs["manifest_sha256"])
        expected_normalised_sha = str(
            expected_outputs["normalised_manifest_sha256"]
        )
        manifest_semantic_equal = (
            normalised_rebuilt_sha == expected_normalised_sha
        )
    else:
        frozen_manifest_path = canonical / "manifest.json"
        frozen_manifest = json.loads(
            frozen_manifest_path.read_text(encoding="utf-8")
        )
        frozen_manifest_hash = sha256(frozen_manifest_path)
        expected_normalised_sha = _json_object_sha256(
            _normalise_evaluation_manifest_provenance(frozen_manifest)
        )
        manifest_semantic_equal = (
            normalised_rebuilt_sha == expected_normalised_sha
        )
    comparison["manifest.json"] = {
        "rebuilt_sha256": sha256(rebuilt_manifest_path),
        "expected_sha256": frozen_manifest_hash,
        "byte_identical": (
            sha256(rebuilt_manifest_path) == frozen_manifest_hash
        ),
        "normalised_rebuilt_sha256": normalised_rebuilt_sha,
        "normalised_expected_sha256": expected_normalised_sha,
        "semantic_equal_after_source_path_normalisation": (
            manifest_semantic_equal
        ),
    }
    rebuilt_sums = json.loads(
        (out_dir / "SHA256SUMS.json").read_text(encoding="utf-8")
    )
    rebuilt_non_manifest_sums = {
        key: value for key, value in rebuilt_sums.items()
        if key != "manifest.json"
    }
    if expected_outputs is not None:
        expected_non_manifest_sums = {
            key: value
            for key, value in expected_outputs["files"].items()
            if key in rebuilt_non_manifest_sums
        }
        checksum_payload_equal = (
            rebuilt_non_manifest_sums == expected_non_manifest_sums
        )
        expected_sums_hash = str(expected_outputs["sha256sums_sha256"])
    else:
        frozen_sums = json.loads(
            (canonical / "SHA256SUMS.json").read_text(encoding="utf-8")
        )
        expected_non_manifest_sums = {
            key: value for key, value in frozen_sums.items()
            if key != "manifest.json"
        }
        checksum_payload_equal = (
            rebuilt_non_manifest_sums == expected_non_manifest_sums
        )
        expected_sums_hash = sha256(canonical / "SHA256SUMS.json")
    comparison["SHA256SUMS.json"] = {
        "rebuilt_sha256": sha256(out_dir / "SHA256SUMS.json"),
        "expected_sha256": expected_sums_hash,
        "byte_identical": (
            sha256(out_dir / "SHA256SUMS.json")
            == expected_sums_hash
        ),
        "non_manifest_checksums_equal": checksum_payload_equal,
    }
    byte_outputs_equal = all(
        comparison[name]["byte_identical"] for name in byte_names
    )
    if not (
        byte_outputs_equal
        and manifest_semantic_equal
        and checksum_payload_equal
    ):
        mismatches = [
            name
            for name, item in comparison.items()
            if not item["byte_identical"]
        ]
        raise ValueError(f"evaluation rebuild differs from canonical: {mismatches}")
    return {
        "status": "passed",
        "out_dir": str(out_dir),
        "files": comparison,
        "stdout_suppressed_to_avoid_test_metadata_dump": bool(completed.stdout),
        "reranker_split_stdout_suppressed": bool(split_completed.stdout),
    }


def _single_multi_command(
    cfg: dict,
    phase: str,
    out_dir: Path,
    *,
    queries: Path,
    labels: Path,
) -> list[str]:
    spec = cfg["paper_evaluation_cohorts"]["balanced_single_multi400"]
    command = [
        sys.executable,
        str(resolve(spec["script"])),
        "--phase",
        phase,
        "--queries",
        str(queries),
        "--labels",
        str(labels),
        "--out-dir",
        str(out_dir),
        "--seed",
        str(spec["seed"]),
    ]
    if phase == "prepare":
        command.extend([
            "--candidate-per-label",
            str(spec["confirmation_candidate_per_label"]),
        ])
    elif phase == "final":
        command.extend(["--per-label", str(spec["per_label"])])
    else:
        raise ValueError(f"unsupported single/multi rebuild phase: {phase}")
    return command


def rebuild_paper_evaluation_cohorts(cfg: dict, out_dir: Path) -> dict:
    """Rebuild the seeded 400→398/200 thesis cohorts around frozen LLM labels."""
    structure_dir = out_dir / "single_multi_400"
    evaluation_dir = out_dir / "natural_query_eval_splits_v1"
    structure_dir.mkdir(parents=True, exist_ok=True)
    spec = cfg["paper_evaluation_cohorts"]["balanced_single_multi400"]

    _run(_single_multi_command(
        cfg,
        "prepare",
        structure_dir,
        queries=resolve(cfg["evaluation_freeze"]["eligible_unseen"]),
        labels=resolve(spec["pass1_labels"]),
    ))
    _run(_single_multi_command(
        cfg,
        "final",
        structure_dir,
        queries=structure_dir / "confirmation_queries.csv",
        labels=resolve(spec["confirmation_labels"]),
    ))

    canonical_structure = resolve(spec["output"]).parent
    structure_files = (
        "confirmation_queries.csv",
        "manifest_prepare.json",
        "single_multi_queries_400_ADMIN.csv",
        "single_need_queries.csv",
        "multi_need_queries.csv",
        "query_structure_review_blind.csv",
        "manifest_final.json",
    )
    structure_comparison = {}
    for name in structure_files:
        rebuilt = structure_dir / name
        canonical = canonical_structure / name
        structure_comparison[name] = {
            "rebuilt_sha256": sha256(rebuilt),
            "canonical_sha256": sha256(canonical),
            "byte_identical": sha256(rebuilt) == sha256(canonical),
        }
    if not all(
        row["byte_identical"] for row in structure_comparison.values()
    ):
        raise ValueError("single/multi 400 cohort is not byte reproducible")

    canonical_exclusions = [
        resolve(path)
        for path in cfg["evaluation_freeze"]["test_exclusion_files"]
    ]
    canonical_confirmation = resolve(spec["confirmation_candidates"])
    rebuilt_exclusions = [
        structure_dir / "confirmation_queries.csv"
        if path == canonical_confirmation
        else path
        for path in canonical_exclusions
    ]
    evaluation = rebuild_evaluation(
        cfg,
        evaluation_dir,
        single_multi_library=(
            structure_dir / "single_multi_queries_400_ADMIN.csv"
        ),
        test_exclusion_files=rebuilt_exclusions,
    )
    return {
        "status": "passed",
        "frozen_hosted_label_boundary": {
            "pass1_labels": str(resolve(spec["pass1_labels"])),
            "confirmation_labels": str(resolve(spec["confirmation_labels"])),
            "hosted_model_recalled": False,
        },
        "single_multi400": structure_comparison,
        "development398_and_test200": evaluation,
    }


def open_source_bundle_inventory(cfg: dict) -> dict:
    """Describe the portable processed-input bundle without copying text."""
    spec = cfg["open_source_partition_reproduction"]
    mixed = cfg["graph_build_corpus"]["historical_mixed_split"]
    paper = cfg["paper_evaluation_cohorts"]["balanced_single_multi400"]
    evaluation = cfg["evaluation_freeze"]
    return {
        "role": spec["role"],
        "warning": spec["warning"],
        "bundle_inputs": {
            name: {
                key: value
                for key, value in item.items()
                if key != "canonical_path"
            }
            for name, item in spec["bundle_inputs"].items()
        },
        "seeds": {
            "historical_600_200_1142": mixed["seed"],
            "natural_query_gate": spec["natural_query_gate"]["seed"],
            "confirmation600_and_balanced400": paper["seed"],
            "development398": evaluation["development_seed"],
            "confirmatory_test200": evaluation["test_seed"],
            "reranker_grouped_cv": evaluation["reranker_cv"]["seeds"],
        },
        "current_outputs": {
            "construction": [600, 200, 1142],
            "eligible_unseen": 1137,
            "confirmation_candidates": 600,
            "balanced_library": {
                "single_need": 200,
                "multi_need": 200,
            },
            "development": {
                "single_need": 199,
                "multi_need": 199,
                "total": 398,
            },
            "confirmatory_cross_thread_test": 200,
        },
        "rebuild_command": (
            "PYTHONPATH=.. python data_preparation/sampling/freeze_research_data_partitions.py "
            "rebuild-open-source-partitions --bundle-root /path/to/bundle"
        ),
        "proposed_valid400_repair": spec["proposed_valid400_repair"],
    }


def _json_object_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _portable_expected_outputs(cfg: dict) -> dict:
    """Hash the canonical outputs once for inclusion in a portable bundle."""
    mixed_dir = resolve(
        cfg["graph_build_corpus"]["historical_mixed_split"]["output_dir"]
    )
    natural_dir = resolve(cfg["evaluation_freeze"]["eligible_unseen"]).parent
    paper = cfg["paper_evaluation_cohorts"]["balanced_single_multi400"]
    structure_dir = resolve(paper["output"]).parent
    evaluation_dir = resolve(cfg["evaluation_freeze"]["canonical_output_dir"])

    def files(directory: Path, names: tuple[str, ...]) -> dict[str, str]:
        return {name: sha256(directory / name) for name in names}

    evaluation_files = files(
        evaluation_dir,
        (
            "development_queries_398_ADMIN.csv",
            "development_queries_398.csv",
            "development_queries_398_official.json",
            "cross_thread_test_queries_200_ADMIN.csv",
            "cross_thread_test_ids_200.csv",
            "reranker_grouped_query_split_manifest.json",
        ),
    )
    evaluation_manifest = json.loads(
        (evaluation_dir / "manifest.json").read_text(encoding="utf-8")
    )
    return {
        "historical_construction_split": files(
            mixed_dir,
            (
                "validation_queries.csv",
                "test_queries.csv",
                "reserve_queries.csv",
                "all_mixed_query_candidates.csv",
            ),
        ),
        "natural_query_gate": files(
            natural_dir,
            (
                "eligible_unseen_queries.csv",
                "excluded_same_post_overlap.csv",
                "prelabel_queries.csv",
                "pilot_queries_ADMIN.csv",
            ),
        ),
        "single_multi400": files(
            structure_dir,
            (
                "confirmation_queries.csv",
                "manifest_prepare.json",
                "single_multi_queries_400_ADMIN.csv",
                "single_need_queries.csv",
                "multi_need_queries.csv",
                "query_structure_review_blind.csv",
                "manifest_final.json",
            ),
        ),
        "dev100_v1": files(
            resolve("out/section17_dev_100"),
            ("section17_dev_queries_ADMIN.csv",),
        ),
        "dev100_v2": files(
            resolve("out/section17_dev_100_v2"),
            ("section17_dev_queries_ADMIN.csv",),
        ),
        "evaluation": {
            "files": evaluation_files,
            "manifest_sha256": sha256(evaluation_dir / "manifest.json"),
            "normalised_manifest_sha256": _json_object_sha256(
                _normalise_evaluation_manifest_provenance(
                    evaluation_manifest
                )
            ),
            "sha256sums_sha256": sha256(
                evaluation_dir / "SHA256SUMS.json"
            ),
        },
    }


def _open_source_input(
    cfg: dict,
    name: str,
    bundle_root: Path | None,
) -> Path:
    item = cfg["open_source_partition_reproduction"]["bundle_inputs"][name]
    path = (
        bundle_root / item["bundle_path"]
        if bundle_root is not None
        else resolve(item["canonical_path"])
    )
    if not path.exists():
        raise FileNotFoundError(f"missing open-source bundle input: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
    missing = set(item["required_fields"]) - fields
    if missing:
        raise ValueError(f"{name} missing required fields: {sorted(missing)}")
    if item["verification"] == "sha256":
        actual = sha256(path)
        if actual != item["sha256"]:
            raise ValueError(
                f"{name} SHA-256 mismatch: expected {item['sha256']}, got {actual}"
            )
    elif item["verification"] == "id_set":
        rows, ids = read_ids(path, str(item["id_field"]))
        actual = id_set_sha256(ids)
        if (
            len(ids) != int(item["unique_ids"])
            or actual != item["id_set_sha256"]
        ):
            raise ValueError(
                f"{name} ID-set mismatch: rows={rows}, unique={len(ids)}, "
                f"id_set_sha256={actual}"
            )
    else:
        raise ValueError(f"unsupported bundle verification: {item['verification']}")
    return path


def export_open_source_partition_bundle(
    cfg: dict,
    out_dir: Path,
) -> dict:
    """Materialise the six-input portable bundle without publishing it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exported = {}
    for name, item in cfg["open_source_partition_reproduction"][
        "bundle_inputs"
    ].items():
        source = _open_source_input(cfg, name, None)
        destination = out_dir / item["bundle_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        projection = item.get("release_projection")
        if projection:
            with source.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            projected = {
                tuple(str(row.get(field) or "").strip() for field in projection)
                for row in rows
            }
            projected.discard(tuple("" for _ in projection))
            with destination.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(projection)
                writer.writerows(sorted(projected))
        else:
            shutil.copyfile(source, destination)
        # Re-validate through the portable path, not the canonical source path.
        _open_source_input(cfg, name, out_dir)
        exported[name] = {
            "path": item["bundle_path"],
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
        if item["verification"] == "id_set":
            _, ids = read_ids(destination, str(item["id_field"]))
            exported[name].update({
                "unique_ids": len(ids),
                "id_set_sha256": id_set_sha256(ids),
            })

    manifest = {
        **open_source_bundle_inventory(cfg),
        "exported_files": exported,
        "expected_outputs": _portable_expected_outputs(cfg),
        "contains_reddit_post_text": True,
        "published_or_uploaded": False,
        "release_review_required": (
            "Review dataset licence/terms, ethics approval, participant "
            "privacy, and de-identification before public distribution."
        ),
    }
    manifest_path = out_dir / "partition_bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "status": "materialised_locally_not_published",
        "out_dir": str(out_dir),
        "files": exported,
        "manifest": str(manifest_path),
        "release_review_required": manifest["release_review_required"],
    }


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _latest_label_pass(rows: list[dict], pass_id: int) -> dict[str, dict]:
    """Match the resumable labeler's last-row-wins repair semantics."""
    selected: dict[str, dict] = {}
    for row in rows:
        if int(row["pass_id"]) == pass_id:
            selected[str(row["query_id"])] = row
    return selected


def _json_list(value: object) -> list:
    if isinstance(value, list):
        return value
    text = str(value or "").strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("query-structure list field is not a JSON array")
    return parsed


def _structure_split_assignments(
    rows: list[dict],
    *,
    seed: int,
    second_stage_seed_offset: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, str]:
    """Create a standard seeded split, stratified only by the task label."""
    from sklearn.model_selection import train_test_split

    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("query-structure split fractions must sum to one")
    ordered = sorted(rows, key=lambda row: str(row["query_id"]))
    query_ids = [str(row["query_id"]) for row in ordered]
    labels = [str(row["final_structure_label"]) for row in ordered]
    if not query_ids or len(query_ids) != len(set(query_ids)):
        raise ValueError("query-structure split requires unique non-empty query IDs")
    if any(not label for label in labels):
        raise ValueError("query-structure split requires a final label for every row")

    train_ids, remainder_ids, _, remainder_labels = train_test_split(
        query_ids,
        labels,
        train_size=train_fraction,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    relative_test_fraction = test_fraction / (
        validation_fraction + test_fraction
    )
    validation_ids, test_ids = train_test_split(
        remainder_ids,
        test_size=relative_test_fraction,
        random_state=seed + second_stage_seed_offset,
        shuffle=True,
        stratify=remainder_labels,
    )
    assignments = {
        **{query_id: "train" for query_id in train_ids},
        **{query_id: "validation" for query_id in validation_ids},
        **{query_id: "test" for query_id in test_ids},
    }
    if set(assignments) != set(query_ids):
        raise AssertionError("query-structure split is incomplete")
    return assignments


def _write_dataset_table(base: Path, rows: list[dict]) -> dict:
    """Write the same ordered table as CSV, nested JSONL, and Parquet."""
    import pandas as pd

    ordered = sorted(rows, key=lambda row: str(row["query_id"]))
    if not ordered:
        raise ValueError(f"cannot write empty dataset table: {base}")
    base.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ordered[0])
    csv_path = base.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ordered:
            writer.writerow({
                field: (
                    json.dumps(row[field], ensure_ascii=False, separators=(",", ":"))
                    if isinstance(row[field], (list, dict))
                    else row[field]
                )
                for field in fields
            })

    jsonl_path = base.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )

    parquet_path = base.with_suffix(".parquet")
    pd.DataFrame(ordered, columns=fields).to_parquet(parquet_path, index=False)
    return {
        path.suffix.lstrip("."): {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in (csv_path, jsonl_path, parquet_path)
    }


def _distribution(rows: list[dict], field: str) -> dict[str, int]:
    return dict(sorted(Counter(
        str(row.get(field) or "<missing>") for row in rows
    ).items()))


def _official_query_id(row: dict[str, Any]) -> str:
    raw = str(row.get("query_id") or row.get("id") or "").strip()
    name = Path(raw).name
    return name[:-5] if name.endswith(".json") else name


def _select_stratified_query_ids(
    rows: list[dict],
    *,
    label_field: str,
    train_size: int,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Seeded equal-probability sample, stratified only by a frozen label."""
    from sklearn.model_selection import train_test_split

    ordered = sorted(rows, key=lambda row: str(row["query_id"]))
    query_ids = [str(row["query_id"]) for row in ordered]
    labels = [str(row[label_field]) for row in ordered]
    if not query_ids or len(query_ids) != len(set(query_ids)):
        raise ValueError("stratified sample requires unique non-empty query IDs")
    if any(not label for label in labels):
        raise ValueError("stratified sample requires a label for every query")
    selected, excluded = train_test_split(
        query_ids,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return set(selected), set(excluded)


def freeze_development300(
    cfg: dict,
    out_dir: Path,
    *,
    development_admin: Path | None = None,
    development_official: Path | None = None,
    anchor_official: Path | None = None,
) -> dict:
    """Freeze dev300 as the existing dev100 plus a seeded balanced dev200."""
    import sklearn

    spec = cfg["paper_evaluation_cohorts"]["development300"]
    sources = {
        "development_admin": development_admin or resolve(
            spec["development_admin_file"]
        ),
        "development_official": development_official or resolve(
            spec["development_official_file"]
        ),
        "anchor_official": anchor_official or resolve(
            spec["anchor_query_file"]
        ),
    }
    for name, path in sources.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: missing development300 source: {path}")

    admin_rows = _read_csv_rows(sources["development_admin"])
    admin_by_id = {str(row["query_id"]): row for row in admin_rows}
    if len(admin_by_id) != len(admin_rows):
        raise ValueError("development398 ADMIN contains duplicate query IDs")
    official_rows = json.loads(
        sources["development_official"].read_text(encoding="utf-8")
    )
    anchor_rows = json.loads(
        sources["anchor_official"].read_text(encoding="utf-8")
    )
    if not isinstance(official_rows, list) or not isinstance(anchor_rows, list):
        raise ValueError("development official inputs must be JSON arrays")
    official_by_id = {_official_query_id(row): row for row in official_rows}
    anchor_ids = {_official_query_id(row) for row in anchor_rows}
    development_ids = set(admin_by_id)
    if len(official_by_id) != len(official_rows):
        raise ValueError("development398 official input has duplicate query IDs")
    if set(official_by_id) != development_ids:
        raise ValueError("development398 ADMIN and official query IDs differ")
    if len(anchor_ids) != int(spec["anchor_query_count"]):
        raise ValueError("nested development100 anchor count differs from config")
    if not anchor_ids <= development_ids:
        raise ValueError("nested development100 is outside development398")

    remaining_rows = [
        row for qid, row in admin_by_id.items() if qid not in anchor_ids
    ]
    label_field = str(spec["stratify_by"])
    selected_ids, excluded_ids = _select_stratified_query_ids(
        remaining_rows,
        label_field=label_field,
        train_size=int(spec["additional_query_count"]),
        seed=int(spec["seed"]),
    )
    final_ids = anchor_ids | selected_ids
    expected_counts = {
        "development398": 398,
        "anchor100": int(spec["anchor_query_count"]),
        "remaining298": 298,
        "additional200": int(spec["additional_query_count"]),
        "excluded98": 98,
        "development300": int(spec["final_query_count"]),
    }
    actual_counts = {
        "development398": len(development_ids),
        "anchor100": len(anchor_ids),
        "remaining298": len(remaining_rows),
        "additional200": len(selected_ids),
        "excluded98": len(excluded_ids),
        "development300": len(final_ids),
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"development300 count contract failed: {actual_counts}"
        )
    if selected_ids & excluded_ids or selected_ids & anchor_ids:
        raise ValueError("development300 components overlap")
    if selected_ids | excluded_ids != development_ids - anchor_ids:
        raise ValueError("development300 selection does not partition remaining298")

    additional_admin = [admin_by_id[qid] for qid in sorted(selected_ids)]
    final_admin = [admin_by_id[qid] for qid in sorted(final_ids)]
    anchor_admin = [admin_by_id[qid] for qid in sorted(anchor_ids)]
    label_counts = {
        "anchor100": _distribution(anchor_admin, label_field),
        "remaining298": _distribution(remaining_rows, label_field),
        "additional200": _distribution(additional_admin, label_field),
        "development300": _distribution(final_admin, label_field),
    }
    target_additional = int(spec["per_label_additional"])
    target_final = int(spec["per_label_final"])
    for label in ("single_need", "multi_need"):
        if label_counts["additional200"].get(label) != target_additional:
            raise ValueError("additional200 is not exactly single/multi balanced")
        if label_counts["development300"].get(label) != target_final:
            raise ValueError("development300 is not exactly single/multi balanced")

    out_dir.mkdir(parents=True, exist_ok=False)
    additional_official = [official_by_id[qid] for qid in sorted(selected_ids)]
    final_official = [official_by_id[qid] for qid in sorted(final_ids)]
    (out_dir / "additional200_queries_official.json").write_text(
        json.dumps(additional_official, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "development300_queries_official.json").write_text(
        json.dumps(final_official, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    safe_fields = (
        "split", "query_id", "condition", "query_text", "structure_label"
    )
    with (out_dir / "additional200_queries.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=safe_fields)
        writer.writeheader()
        for row in additional_admin:
            label = str(row[label_field])
            writer.writerow({
                "split": "development",
                "query_id": row["query_id"],
                "condition": (
                    "unseen_single_need" if label == "single_need"
                    else "unseen_multi_need"
                ),
                "query_text": row["query_text"],
                "structure_label": label,
            })

    admin_fields = list(admin_rows[0]) + [
        "development300_component", "development300_seed"
    ]
    with (out_dir / "development300_queries_ADMIN.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=admin_fields)
        writer.writeheader()
        for row in final_admin:
            writer.writerow({
                **row,
                "development300_component": (
                    "nested_development100" if row["query_id"] in anchor_ids
                    else "seeded_additional200"
                ),
                "development300_seed": spec["seed"],
            })
    (out_dir / "excluded_remaining98_query_ids.txt").write_text(
        "".join(f"{qid}\n" for qid in sorted(excluded_ids)),
        encoding="utf-8",
    )

    distributions = {
        cohort: {
            field: _distribution(rows, field)
            for field in (label_field, "scenario", "tier", "source", "post_role")
        }
        for cohort, rows in {
            "remaining298": remaining_rows,
            "additional200": additional_admin,
            "development300": final_admin,
        }.items()
    }
    proportional_audit = {}
    for field in ("scenario", "tier", "source", "post_role"):
        remaining = distributions["remaining298"][field]
        selected = distributions["additional200"][field]
        rows = {}
        for value, population_n in remaining.items():
            expected = population_n * len(selected_ids) / len(remaining_rows)
            observed = selected.get(value, 0)
            rows[value] = {
                "population_n": population_n,
                "expected_under_proportional_sampling": expected,
                "observed_n": observed,
                "observed_minus_expected": observed - expected,
            }
        proportional_audit[field] = rows

    output_names = (
        "additional200_queries_official.json",
        "development300_queries_official.json",
        "additional200_queries.csv",
        "development300_queries_ADMIN.csv",
        "excluded_remaining98_query_ids.txt",
    )
    manifest = {
        "version": "development300-stratified-v1",
        "status": "FROZEN_LOCAL_QUERY_ONLY",
        "selection": {
            "seed": int(spec["seed"]),
            "method": "sklearn.model_selection.train_test_split",
            "shuffle": True,
            "stratify_by": label_field,
            "selection_frame": "development398_minus_nested_development100",
            "selection_type": spec["selection_type"],
            "equal_probability_within_each_structure_stratum": True,
            "scenario_tier_source_control_selection": False,
            "retrieval_or_utility_control_selection": False,
        },
        "sources": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in sources.items()
        },
        "counts": actual_counts,
        "label_counts": label_counts,
        "distributions": distributions,
        "post_hoc_proportional_distribution_audit": proportional_audit,
        "id_set_sha256": {
            "anchor100": id_set_sha256(anchor_ids),
            "additional200": id_set_sha256(selected_ids),
            "excluded98": id_set_sha256(excluded_ids),
            "development300": id_set_sha256(final_ids),
        },
        "invariants": {
            "development300_subset_of_development398": final_ids <= development_ids,
            "anchor_additional_excluded_pairwise_disjoint": not (
                anchor_ids & selected_ids or anchor_ids & excluded_ids
                or selected_ids & excluded_ids
            ),
            "remaining298_fully_partitioned": (
                selected_ids | excluded_ids == development_ids - anchor_ids
            ),
            "additional_and_final_exactly_balanced": True,
            "confirmatory_test_read": False,
            "retrieval_outputs_read": False,
            "utility_judgments_read": False,
            "external_calls": 0,
        },
        "software": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
        },
        "outputs": {
            name: {
                "path": str(out_dir / name),
                "bytes": (out_dir / name).stat().st_size,
                "sha256": sha256(out_dir / name),
            }
            for name in output_names
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _query_structure_dataset_card(
    cfg: dict,
    *,
    frame_n: int,
    library_n: int,
    development_n: int,
    split_counts: dict[str, int],
) -> str:
    spec = cfg["query_structure_dataset_release"]
    split = spec["structure_classifier_split"]
    return f"""---
language:
- en
task_categories:
- text-classification
pretty_name: ADHD Peer-Support Query Structure
size_categories:
- n<1K
license: other
---

# ADHD Peer-Support Query Structure Dataset

This local, unpublished dataset package contains {library_n} balanced
LLM-silver query-structure records drawn from a {frame_n}-query eligible
sampling frame. The current valid development cohort contains {development_n}
queries.

## Files

- `metadata/sampling_frame_1137.*`: all eligible queries, available label
  passes, membership flags, and upstream strata.
- `metadata/labeled_balanced400.*`: the frozen 200 single-need + 200
  multi-need library, including the two records outside development398.
- `metadata/development398.*`: the valid 199 + 199 development cohort.
- `data/train.*`, `data/validation.*`, `data/test.*`: standard
  query-structure classification splits in CSV, JSONL, and Parquet.
- `splits/assignments.csv` and `splits/ids.json`: compact reproducible split
  membership.
- `manifest.json`: input/output hashes, label protocol, distributions, seed,
  software versions, and role boundaries.

## Label schema

`final_structure_label` is `single_need` or `multi_need`. It is accepted only
after two query-only model passes agree and both exact-span traceability checks
pass. `request_atoms` are independently satisfiable requests grounded by spans
copied from the source post; `constraints` modify a request and do not create a
new need. These labels are LLM silver, not human gold.

## Where the original strata came from

- `source`: whether the query came from the original 1,105-post branch or the
  later 4x expanded branch.
- `scenario`: the frozen primary ADHD-life scenario. The expanded branch used
  an all-MiniLM-L6-v2 prototype classifier trained from the frozen
  LLM-annotated reference set.
- `tier`: the number of qualifying top-level replies: shallow=1--3,
  mid=4--15, deep=16+.
- `original_sampling_stratum`: `scenario::tier::source`, the historical
  sampling key.

All these fields are retained for provenance and distribution audits. They are
not all forced into the new classifier split.

## Train/validation/test split

The standard classifier split uses scikit-learn `train_test_split` twice with
`random_state={split["random_state"]}` and
`random_state={int(split["random_state"]) + int(split["second_stage_random_state_offset"])}`.
It is stratified only by `final_structure_label`, producing
train={split_counts["train"]}, validation={split_counts["validation"]}, and
test={split_counts["test"]}. This deliberately avoids hand-built combinations
of scenario, tier, and source.

The file named `test` is an **internal query-structure classifier test**. It is
not the retrieval system's untouched confirmatory test. The separately frozen
cross-thread test200 remains the only system-level confirmatory set.

## Formats

CSV is intended for inspection and spreadsheets. JSONL preserves request atoms
and constraints as nested JSON arrays. Parquet is the preferred typed format
for pandas, Arrow, and Hugging Face Datasets.

## Responsible release boundary

The files contain Reddit text that may include sensitive lived-experience
content. This package is materialised locally and is not approved for public
upload. Licence/terms, ethics approval, privacy, and de-identification must be
reviewed before publication.
"""


def materialize_query_structure_dataset(
    cfg: dict,
    out_dir: Path,
    *,
    sampling_frame: Path | None = None,
    pass_labels: Path | None = None,
    confirmation_queries: Path | None = None,
    confirmation_labels: Path | None = None,
    balanced_library: Path | None = None,
    development: Path | None = None,
    system_confirmatory_test_ids: Path | None = None,
) -> dict:
    """Materialise a standard labelled table without changing frozen cohorts."""
    import pandas as pd
    import pyarrow
    import sklearn

    spec = cfg["query_structure_dataset_release"]
    inputs = spec["inputs"]
    source_paths = {
        "sampling_frame": sampling_frame or resolve(inputs["sampling_frame"]),
        "pass_labels": pass_labels or resolve(inputs["pass_labels"]),
        "confirmation_queries": (
            confirmation_queries or resolve(inputs["confirmation_queries"])
        ),
        "confirmation_labels": (
            confirmation_labels or resolve(inputs["confirmation_labels"])
        ),
        "balanced_library": (
            balanced_library or resolve(inputs["balanced_library"])
        ),
        "development": development or resolve(inputs["development"]),
        "system_confirmatory_test_ids": (
            system_confirmatory_test_ids
            or resolve(inputs["system_confirmatory_test_ids"])
        ),
        "label_prompt": resolve(inputs["label_prompt"]),
    }
    for name, path in source_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: missing dataset input: {path}")

    frame_rows = _read_csv_rows(source_paths["sampling_frame"])
    frame_by_id = {str(row["query_id"]): row for row in frame_rows}
    if len(frame_by_id) != len(frame_rows):
        raise ValueError("sampling frame contains duplicate query IDs")
    label_rows = _read_csv_rows(source_paths["pass_labels"])
    confirmation_label_rows = _read_csv_rows(
        source_paths["confirmation_labels"]
    )
    confirmation_ids = {
        str(row["query_id"])
        for row in _read_csv_rows(source_paths["confirmation_queries"])
    }
    library_rows = _read_csv_rows(source_paths["balanced_library"])
    library_by_id = {str(row["query_id"]): row for row in library_rows}
    development_rows = _read_csv_rows(source_paths["development"])
    development_ids = {str(row["query_id"]) for row in development_rows}
    system_test_rows = _read_csv_rows(
        source_paths["system_confirmatory_test_ids"]
    )
    system_test_by_id = {
        str(row["query_id"]): row for row in system_test_rows
    }
    invalid_ids = set(
        cfg["paper_evaluation_cohorts"]["development398"][
            "excluded_query_ids"
        ]
    )

    initial_p1 = _latest_label_pass(label_rows, 1)
    initial_p2 = _latest_label_pass(label_rows, 2)
    confirmation_p1 = _latest_label_pass(confirmation_label_rows, 1)
    confirmation_p2 = _latest_label_pass(confirmation_label_rows, 2)

    master_rows = []
    for query_id in sorted(frame_by_id):
        source = frame_by_id[query_id]
        if query_id in confirmation_ids:
            pass1 = confirmation_p1.get(query_id)
            pass2 = confirmation_p2.get(query_id)
        else:
            pass1 = initial_p1.get(query_id)
            pass2 = initial_p2.get(query_id)
        library = library_by_id.get(query_id)
        pass1_trace = int(pass1["traceability_pass"]) if pass1 else None
        pass2_trace = int(pass2["traceability_pass"]) if pass2 else None
        agreement = (
            pass1["single_multi_label"] == pass2["single_multi_label"]
            if pass1 and pass2 else None
        )
        agreed_traceable = bool(
            pass1
            and pass2
            and agreement
            and pass1_trace == pass2_trace == 1
        )
        if library:
            final_label = str(library["llm_single_multi_label"])
            final_need_count = int(library["llm_need_count"])
            final_atoms = _json_list(library["llm_request_atoms_json"])
            final_constraints = _json_list(library["llm_constraints_json"])
            if not agreed_traceable or final_label != pass1["single_multi_label"]:
                raise ValueError(
                    f"balanced library label is not two-pass agreed: {query_id}"
                )
            label_status = "agreed_traceable_balanced_library"
        elif agreed_traceable:
            final_label = str(pass1["single_multi_label"])
            final_need_count = int(pass1["need_count"])
            final_atoms = _json_list(pass1["request_atoms_json"])
            final_constraints = _json_list(pass1["constraints_json"])
            label_status = "agreed_traceable_not_selected"
        else:
            final_label = ""
            final_need_count = None
            final_atoms = []
            final_constraints = []
            if pass1 and pass2 and not agreement:
                label_status = "two_pass_disagreement"
            elif pass1 and pass2:
                label_status = "traceability_failure"
            elif pass1:
                label_status = "pass1_only"
            else:
                label_status = "unlabelled"

        upstream_stratum = "::".join((
            str(source.get("scenario") or "UNKNOWN"),
            str(source.get("tier") or "UNKNOWN"),
            str(source.get("source") or "UNKNOWN"),
        ))
        system_test = system_test_by_id.get(query_id)
        in_development = query_id in development_ids
        in_library = library is not None
        if in_development and system_test:
            raise ValueError("development and system confirmatory test overlap")
        if in_development:
            retrieval_role = "development"
        elif system_test:
            retrieval_role = "confirmatory_cross_thread_test"
        elif in_library:
            retrieval_role = "balanced_library_only"
        else:
            retrieval_role = "sampling_frame_or_reserve"

        split_exclusion = ""
        if in_library and not in_development:
            split_exclusion = (
                "excluded_no_identifiable_need"
                if query_id in invalid_ids
                else "balanced_1_to_1_truncation"
            )
        row = {
            "dataset_version": spec["version"],
            "query_id": query_id,
            "post_id": str(source.get("post_id") or query_id),
            "query_text": str(source.get("query_text") or ""),
            "source": str(source.get("source") or ""),
            "scenario": str(source.get("scenario") or ""),
            "tier": str(source.get("tier") or ""),
            "original_sampling_stratum": upstream_stratum,
            "created_year": str(source.get("created_year") or ""),
            "post_role": str(source.get("post_role") or ""),
            "role_confidence": (
                float(source["role_confidence"])
                if str(source.get("role_confidence") or "").strip()
                else None
            ),
            "role_reason": str(source.get("role_reason") or ""),
            "role_label_provenance": "keyword_heuristic_v1",
            "drug_related": bool(int(source.get("drug_related") or 0)),
            "drug_match": str(source.get("drug_match") or ""),
            "pass1_label": str(pass1.get("single_multi_label") or "") if pass1 else "",
            "pass1_need_count": int(pass1["need_count"]) if pass1 else None,
            "pass1_request_atoms": (
                _json_list(pass1["request_atoms_json"]) if pass1 else []
            ),
            "pass1_constraints": (
                _json_list(pass1["constraints_json"]) if pass1 else []
            ),
            "pass1_traceability_pass": pass1_trace,
            "pass2_label": str(pass2.get("single_multi_label") or "") if pass2 else "",
            "pass2_need_count": int(pass2["need_count"]) if pass2 else None,
            "pass2_request_atoms": (
                _json_list(pass2["request_atoms_json"]) if pass2 else []
            ),
            "pass2_constraints": (
                _json_list(pass2["constraints_json"]) if pass2 else []
            ),
            "pass2_traceability_pass": pass2_trace,
            "two_pass_label_agreement": agreement,
            "final_structure_label": final_label,
            "final_need_count": final_need_count,
            "final_request_atoms": final_atoms,
            "final_constraints": final_constraints,
            "label_status": label_status,
            "label_provenance": (
                "query_only_llm_v2_two_pass"
                if final_label else ""
            ),
            "in_confirmation600": query_id in confirmation_ids,
            "in_balanced_library400": in_library,
            "in_current_development398": in_development,
            "in_system_confirmatory_test200": bool(system_test),
            "library_sampling_stratum": (
                f"{final_label}::{upstream_stratum}" if in_library else ""
            ),
            "structure_split": "",
            "structure_split_stratum": (
                final_label if in_development else ""
            ),
            "structure_split_seed": None,
            "structure_split_exclusion_reason": split_exclusion,
            "current_retrieval_role": retrieval_role,
            "system_test_sampling_stratum": (
                str(system_test.get("sampling_stratum") or "")
                if system_test else ""
            ),
            "system_test_stratum_population_n": (
                int(system_test["stratum_population_n"])
                if system_test else None
            ),
            "system_test_stratum_selected_n": (
                int(system_test["stratum_selected_n"])
                if system_test else None
            ),
            "system_test_inclusion_probability": (
                float(system_test["inclusion_probability"])
                if system_test else None
            ),
        }
        master_rows.append(row)

    master_by_id = {row["query_id"]: row for row in master_rows}
    if not development_ids <= set(master_by_id):
        raise ValueError("development398 is outside the sampling frame")
    development_dataset = [master_by_id[qid] for qid in sorted(development_ids)]
    if any(
        row["final_structure_label"] not in {"single_need", "multi_need"}
        for row in development_dataset
    ):
        raise ValueError("development398 contains a missing/non-target label")

    split = spec["structure_classifier_split"]
    assignments = _structure_split_assignments(
        development_dataset,
        seed=int(split["random_state"]),
        second_stage_seed_offset=int(
            split["second_stage_random_state_offset"]
        ),
        train_fraction=float(split["train_fraction"]),
        validation_fraction=float(split["validation_fraction"]),
        test_fraction=float(split["test_fraction"]),
    )
    for query_id, split_name in assignments.items():
        row = master_by_id[query_id]
        row["structure_split"] = split_name
        row["structure_split_seed"] = int(split["random_state"])

    development_dataset = [
        master_by_id[qid] for qid in sorted(development_ids)
    ]
    library_dataset = [
        master_by_id[qid] for qid in sorted(library_by_id)
    ]
    split_rows = {
        split_name: [
            row for row in development_dataset
            if row["structure_split"] == split_name
        ]
        for split_name in ("train", "validation", "test")
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sampling_frame1137": _write_dataset_table(
            out_dir / "metadata" / "sampling_frame_1137", master_rows
        ),
        "labeled_balanced400": _write_dataset_table(
            out_dir / "metadata" / "labeled_balanced400", library_dataset
        ),
        "development398": _write_dataset_table(
            out_dir / "metadata" / "development398", development_dataset
        ),
    }
    for split_name, rows in split_rows.items():
        outputs[split_name] = _write_dataset_table(
            out_dir / "data" / split_name, rows
        )

    assignments_path = out_dir / "splits" / "assignments.csv"
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    with assignments_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "query_id", "structure_split", "split_stratum", "random_state",
        ])
        writer.writeheader()
        for query_id in sorted(assignments):
            writer.writerow({
                "query_id": query_id,
                "structure_split": assignments[query_id],
                "split_stratum": master_by_id[query_id][
                    "final_structure_label"
                ],
                "random_state": split["random_state"],
            })
    split_ids_path = out_dir / "splits" / "ids.json"
    split_ids_path.write_text(
        json.dumps({
            name: sorted(row["query_id"] for row in rows)
            for name, rows in split_rows.items()
        }, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    outputs["split_assignments"] = {
        "csv": {
            "path": str(assignments_path),
            "bytes": assignments_path.stat().st_size,
            "sha256": sha256(assignments_path),
        },
        "json": {
            "path": str(split_ids_path),
            "bytes": split_ids_path.stat().st_size,
            "sha256": sha256(split_ids_path),
        },
    }

    split_counts = {
        name: len(rows) for name, rows in split_rows.items()
    }
    card_path = out_dir / "README.md"
    card_path.write_text(
        _query_structure_dataset_card(
            cfg,
            frame_n=len(master_rows),
            library_n=len(library_dataset),
            development_n=len(development_dataset),
            split_counts=split_counts,
        ),
        encoding="utf-8",
    )
    outputs["dataset_card"] = {
        "md": {
            "path": str(card_path),
            "bytes": card_path.stat().st_size,
            "sha256": sha256(card_path),
        }
    }

    manifest = {
        "dataset_version": spec["version"],
        "status": "materialised_locally_not_published",
        "role": spec["role"],
        "inputs": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in source_paths.items()
        },
        "label_protocol": spec["label_protocol"],
        "upstream_strata": spec["upstream_strata"],
        "structure_classifier_split": {
            **split,
            "counts": split_counts,
            "label_counts": {
                name: _distribution(rows, "final_structure_label")
                for name, rows in split_rows.items()
            },
            "query_id_sets_pairwise_disjoint": (
                not (
                    set(row["query_id"] for row in split_rows["train"])
                    & set(row["query_id"] for row in split_rows["validation"])
                )
                and not (
                    set(row["query_id"] for row in split_rows["train"])
                    & set(row["query_id"] for row in split_rows["test"])
                )
                and not (
                    set(row["query_id"] for row in split_rows["validation"])
                    & set(row["query_id"] for row in split_rows["test"])
                )
            ),
            "overlap_with_system_confirmatory_test200": len(
                set(assignments) & set(system_test_by_id)
            ),
        },
        "counts": {
            "sampling_frame": len(master_rows),
            "confirmation_candidates": len(confirmation_ids),
            "balanced_library": len(library_dataset),
            "development": len(development_dataset),
            "system_confirmatory_test_ids": len(system_test_by_id),
            "label_status": _distribution(master_rows, "label_status"),
        },
        "distributions": {
            field: _distribution(master_rows, field)
            for field in (
                "source", "scenario", "tier", "post_role",
                "final_structure_label", "current_retrieval_role",
            )
        },
        "software": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "outputs": outputs,
        "boundaries": {
            "classifier_test_is_system_confirmatory_test": False,
            "retrieval_system_confirmatory_test": (
                "out/natural_query_eval_splits_v1/"
                "cross_thread_test_ids_200.csv"
            ),
            "labels_are_human_gold": False,
            "contains_reddit_post_text": True,
            "published_or_uploaded": False,
            "release_review_required": (
                "Review dataset licence/terms, ethics approval, privacy, and "
                "de-identification before public distribution."
            ),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "status": manifest["status"],
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
        "counts": manifest["counts"],
        "split": manifest["structure_classifier_split"],
    }


def _compare_rebuilt_files(
    rebuilt_dir: Path,
    canonical_dir: Path | None,
    names: tuple[str, ...],
    label: str,
    *,
    expected_hashes: dict[str, str] | None = None,
) -> dict:
    if (canonical_dir is None) == (expected_hashes is None):
        raise ValueError(
            "provide exactly one of canonical_dir or expected_hashes"
        )
    comparison = {}
    for name in names:
        rebuilt = rebuilt_dir / name
        expected_hash = (
            str(expected_hashes[name])
            if expected_hashes is not None
            else sha256(canonical_dir / name)
        )
        comparison[name] = {
            "rebuilt_sha256": sha256(rebuilt),
            "expected_sha256": expected_hash,
            "byte_identical": sha256(rebuilt) == expected_hash,
        }
    if not all(item["byte_identical"] for item in comparison.values()):
        raise ValueError(f"{label} differs from canonical files")
    return comparison


def _natural_query_gate_command(
    cfg: dict,
    *,
    reserve_queries: Path,
    graph_membership: Path,
    out_dir: Path,
) -> list[str]:
    spec = cfg["open_source_partition_reproduction"]["natural_query_gate"]
    return [
        sys.executable,
        str(resolve(spec["script"])),
        "--queries",
        str(reserve_queries),
        "--corpus-map",
        str(graph_membership),
        "--out-dir",
        str(out_dir),
        "--prelabel-n",
        str(spec["prelabel_n"]),
        "--random-core-n",
        str(spec["random_core_n"]),
        "--single-n",
        str(spec["single_need_enrichment_n"]),
        "--multi-n",
        str(spec["multi_need_enrichment_n"]),
        "--quality-n-per-label",
        str(spec["retrieval_quality_pilot_per_label"]),
        "--seed",
        str(spec["seed"]),
        "--dataset",
        str(spec["dataset"]),
    ]


def rebuild_open_source_partitions(
    cfg: dict,
    out_dir: Path,
    *,
    bundle_root: Path | None = None,
) -> dict:
    """Rebuild the current split lineage from the six portable bundle inputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        name: _open_source_input(cfg, name, bundle_root)
        for name in cfg["open_source_partition_reproduction"]["bundle_inputs"]
    }
    expected = None
    if bundle_root is not None:
        bundle_manifest_path = bundle_root / "partition_bundle_manifest.json"
        if not bundle_manifest_path.exists():
            raise FileNotFoundError(
                "portable bundle is missing partition_bundle_manifest.json"
            )
        bundle_manifest = json.loads(
            bundle_manifest_path.read_text(encoding="utf-8")
        )
        expected = bundle_manifest.get("expected_outputs")
        if not isinstance(expected, dict):
            raise ValueError(
                "portable bundle manifest is missing expected_outputs"
            )

    mixed_dir = out_dir / "mixed_query_splits_800"
    natural_dir = out_dir / "natural_query_benchmark_v1"
    structure_dir = out_dir / "natural_query_single_multi_400"
    dev100_v1_dir = out_dir / "section17_dev_100"
    dev100_v2_dir = out_dir / "section17_dev_100_v2"
    evaluation_dir = out_dir / "natural_query_eval_splits_v1"
    for directory in (
        mixed_dir,
        natural_dir,
        structure_dir,
        dev100_v1_dir,
        dev100_v2_dir,
        evaluation_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    _run(_historical_split_command(
        cfg,
        mixed_dir,
        original_audit=inputs["original_role_audit"],
        expanded_audit=inputs["expanded_role_audit"],
        exclude_paths=[inputs["prior_eval_exclusions"]],
    ))
    mixed_comparison = _compare_rebuilt_files(
        mixed_dir,
        None if expected is not None else resolve(
            cfg["graph_build_corpus"]["historical_mixed_split"]["output_dir"]
        ),
        (
            "validation_queries.csv",
            "test_queries.csv",
            "reserve_queries.csv",
            "all_mixed_query_candidates.csv",
        ),
        "historical 600/200/1142 split",
        expected_hashes=(
            expected["historical_construction_split"]
            if expected is not None else None
        ),
    )

    _run(_natural_query_gate_command(
        cfg,
        reserve_queries=mixed_dir / "reserve_queries.csv",
        graph_membership=inputs["graph_corpus_post_membership"],
        out_dir=natural_dir,
    ))
    natural_comparison = _compare_rebuilt_files(
        natural_dir,
        None if expected is not None else resolve(
            cfg["evaluation_freeze"]["eligible_unseen"]
        ).parent,
        (
            "eligible_unseen_queries.csv",
            "excluded_same_post_overlap.csv",
            "prelabel_queries.csv",
            "pilot_queries_ADMIN.csv",
        ),
        "1142→1137 natural-query gate",
        expected_hashes=(
            expected["natural_query_gate"]
            if expected is not None else None
        ),
    )

    _run(_single_multi_command(
        cfg,
        "prepare",
        structure_dir,
        queries=natural_dir / "eligible_unseen_queries.csv",
        labels=inputs["query_structure_pass_labels"],
    ))
    _run(_single_multi_command(
        cfg,
        "final",
        structure_dir,
        queries=structure_dir / "confirmation_queries.csv",
        labels=inputs["confirmation_labels"],
    ))
    paper = cfg["paper_evaluation_cohorts"]["balanced_single_multi400"]
    structure_comparison = _compare_rebuilt_files(
        structure_dir,
        None if expected is not None else resolve(paper["output"]).parent,
        (
            "confirmation_queries.csv",
            "manifest_prepare.json",
            "single_multi_queries_400_ADMIN.csv",
            "single_need_queries.csv",
            "multi_need_queries.csv",
            "query_structure_review_blind.csv",
            "manifest_final.json",
        ),
        "confirmation600 and balanced400",
        expected_hashes=(
            expected["single_multi400"]
            if expected is not None else None
        ),
    )

    anchor = cfg["open_source_partition_reproduction"]["current_dev_anchor"]
    script = resolve(paper["script"])
    library = structure_dir / "single_multi_queries_400_ADMIN.csv"
    _run([
        sys.executable,
        str(script),
        "--phase",
        "dev",
        "--queries",
        str(library),
        "--out-dir",
        str(dev100_v1_dir),
        "--per-label",
        str(anchor["initial_per_label"]),
        "--seed",
        str(anchor["seed"]),
    ])
    _run([
        sys.executable,
        str(script),
        "--phase",
        "replace",
        "--queries",
        str(library),
        "--base-dev",
        str(dev100_v1_dir / "section17_dev_queries_ADMIN.csv"),
        "--exclude-query-id",
        str(anchor["invalid_query_id"]),
        "--out-dir",
        str(dev100_v2_dir),
        "--seed",
        str(anchor["seed"]),
        "--dataset-version",
        "dev100-v2",
    ])
    replacement_audit = json.loads(
        (dev100_v2_dir / "replacement_audit.json").read_text(encoding="utf-8")
    )
    if replacement_audit["replacement_query_id"] != anchor["replacement_query_id"]:
        raise ValueError("dev100-v2 deterministic replacement changed")
    dev_anchor_comparison = {
        "dev100_v1": _compare_rebuilt_files(
            dev100_v1_dir,
            None if expected is not None else resolve(
                "out/section17_dev_100"
            ),
            ("section17_dev_queries_ADMIN.csv",),
            "dev100-v1 anchor",
            expected_hashes=(
                expected["dev100_v1"] if expected is not None else None
            ),
        ),
        "dev100_v2": _compare_rebuilt_files(
            dev100_v2_dir,
            None if expected is not None else resolve(
                "out/section17_dev_100_v2"
            ),
            ("section17_dev_queries_ADMIN.csv",),
            "dev100-v2 anchor",
            expected_hashes=(
                expected["dev100_v2"] if expected is not None else None
            ),
        ),
        "replacement": replacement_audit,
    }

    evaluation = rebuild_evaluation(
        cfg,
        evaluation_dir,
        single_multi_library=library,
        current_dev=dev100_v2_dir / "section17_dev_queries_ADMIN.csv",
        eligible_unseen=natural_dir / "eligible_unseen_queries.csv",
        test_exclusion_files=[
            natural_dir / "prelabel_queries.csv",
            natural_dir / "pilot_queries_ADMIN.csv",
            structure_dir / "confirmation_queries.csv",
        ],
        audit_splits=[mixed_dir / "test_queries.csv"],
        exposure_audit_files=[inputs["query_structure_pass_labels"]],
        expected_outputs=(
            expected["evaluation"] if expected is not None else None
        ),
    )
    query_structure_dataset = materialize_query_structure_dataset(
        cfg,
        out_dir / "query_structure_dataset_v1",
        sampling_frame=natural_dir / "eligible_unseen_queries.csv",
        pass_labels=inputs["query_structure_pass_labels"],
        confirmation_queries=structure_dir / "confirmation_queries.csv",
        confirmation_labels=inputs["confirmation_labels"],
        balanced_library=library,
        development=(
            evaluation_dir / "development_queries_398_ADMIN.csv"
        ),
        system_confirmatory_test_ids=(
            evaluation_dir / "cross_thread_test_ids_200.csv"
        ),
    )
    return {
        "status": "passed",
        "bundle_mode": bundle_root is not None,
        "canonical_output_files_read": expected is None,
        "hosted_models_recalled": False,
        "input_verification": {
            name: str(path) for name, path in inputs.items()
        },
        "historical_construction_split": mixed_comparison,
        "natural_query_gate": natural_comparison,
        "single_multi400": structure_comparison,
        "current_dev_anchor": dev_anchor_comparison,
        "development398_and_test200": evaluation,
        "query_structure_dataset": query_structure_dataset,
        "proposed_valid400_repair": cfg[
            "open_source_partition_reproduction"
        ]["proposed_valid400_repair"],
    }


def _raw_graph_command(
    cfg: dict, output: Path, manifest: Path
) -> list[str]:
    graph = cfg["raw_graph_sampling"]
    prefix = []
    if graph.get("offline_model_load", False):
        prefix = ["env", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"]
    return prefix + [
        sys.executable, str(resolve(graph["script"])),
        "--dumps-dir", str(resolve(graph["dumps_dir"])),
        "--subreddits", str(graph["subreddit"]),
        "--year-min", str(graph["year_min"]),
        "--year-max", str(graph["year_max"]),
        "--min-post-chars", str(graph["min_post_chars"]),
        "--min-comment-chars", str(graph["min_comment_chars"]),
        "--shallow-q", str(graph["shallow_quota_per_scenario"]),
        "--mid-q", str(graph["mid_quota_per_scenario"]),
        "--deep-q", str(graph["deep_quota_per_scenario"]),
        "--max-per-post", str(graph["max_per_post_non_deep"]),
        "--backend", str(graph["classifier_backend"]),
        "--model", str(graph["classifier_model"]),
        "--model-revision", str(graph["classifier_revision"]),
        "--embedding-batch-size", str(graph["embedding_batch_size"]),
        "--seed", str(graph["seed"]),
        "--text-csv", str(resolve(graph["classifier_reference_texts"])),
        "--post-jsonl", str(resolve(graph["classifier_reference_labels"])),
        "--exclude-jsonl", str(resolve(graph["exclude_posts"])),
        "--out", str(output),
        "--manifest", str(manifest),
    ]


def reproducibility_commands(cfg: dict) -> dict[str, list[str]]:
    graph = cfg["raw_graph_sampling"]
    corpus = cfg["graph_build_corpus"]
    mixed = corpus["historical_mixed_split"]
    unified = corpus["unified_extraction_selection"]
    assembly = corpus["graph_assembly"]
    paper = cfg["paper_evaluation_cohorts"]["balanced_single_multi400"]
    canonical_structure = resolve(paper["output"]).parent
    return {
        "raw_graph_sampling": _raw_graph_command(
            cfg, resolve(graph["output"]), resolve(graph["manifest"])
        ),
        "cap_graph_corpus": [
            sys.executable, str(resolve(corpus["cap_script"])),
            "--input", str(resolve(corpus["source"])),
            "--out-dir", str(resolve(corpus["cap_output_dir"])),
            "--exclude-post-jsonl", str(resolve(corpus["exclude_posts"])),
            "--shallow-cap", str(corpus["shallow_cap"]),
            "--mid-cap", str(corpus["mid_cap"]),
            "--deep-cap", str(corpus["deep_cap"]),
        ],
        "historical_graph_construction_split": [
            sys.executable, str(resolve(mixed["script"])),
            "--original-audit", str(resolve(mixed["original_audit"])),
            "--expanded-audit", str(resolve(mixed["expanded_audit"])),
            "--exclude", str(resolve(mixed["exclude"])),
            "--validation-n", str(mixed["validation_n"]),
            "--test-n", str(mixed["test_n"]),
            "--human-validation-n", str(mixed["human_validation_n"]),
            "--human-test-n", str(mixed["human_test_n"]),
            "--balance", str(mixed["balance"]),
            "--seed", str(mixed["seed"]),
            "--out-dir", str(resolve(mixed["output_dir"])),
        ],
        "unified_graph_extraction_selection": [
            sys.executable, str(resolve(unified["script"])),
            "--old-comments", str(resolve(unified["old_comments"])),
            "--expanded-cap30", str(resolve(unified["expanded_cap30"])),
            "--val", str(resolve(unified["construction_validation"])),
            "--test", str(resolve(unified["construction_test"])),
            "--seed", str(unified["seed"]),
            "--out", str(resolve(unified["output"])),
        ],
        "assemble_project_graph": [
            sys.executable, str(resolve(assembly["build_script"])),
            "--remapped-dir", str(resolve(assembly["remapped_dir"])),
            "--compiled-dir", str(resolve(assembly["compiled_dir"])),
            "--entities-dir", str(resolve(assembly["entities_dir"])),
            "--entity-jsonl", str(resolve(assembly["entity_jsonl"])),
            "--out-dir", str(resolve(assembly["output_dir"])),
        ],
        "densify_project_graph": [
            "env",
            f"STUDY_RETR_BACKEND={assembly['semantic_backend']}",
            f"STUDY_BERT_MODEL={assembly['semantic_model']}",
            f"STUDY_BERT_REVISION={assembly['semantic_model_revision']}",
            f"STUDY_BERT_BATCH_SIZE={assembly['semantic_embedding_batch_size']}",
            f"STUDY_KNN_IMPLEMENTATION={assembly['knn_implementation']}",
            sys.executable, str(resolve(assembly["densify_script"])),
            "--graph", str(resolve(assembly["output_dir"])),
            "--cooc-min", str(assembly["cooc_min"]),
            "--knn-k", str(assembly["knn_k"]),
            "--knn-thresh", str(assembly["knn_thresh"]),
            "--normalize", str(assembly["normalize"]),
            "--relation-weight", str(assembly["relation_weight"]),
            "--cooc-weight", str(assembly["cooc_weight"]),
            "--knn-weight", str(assembly["knn_weight"]),
            "--max-cluster-size", str(assembly["max_cluster_size"]),
            "--seed", str(assembly["leiden_seed"]),
            "--out", str(resolve(assembly["communities"])),
            "--edge-out", str(resolve(assembly["dense_edges"])),
        ],
        "single_multi_confirmation600": _single_multi_command(
            cfg,
            "prepare",
            canonical_structure,
            queries=resolve(cfg["evaluation_freeze"]["eligible_unseen"]),
            labels=resolve(paper["pass1_labels"]),
        ),
        "single_multi_balanced400": _single_multi_command(
            cfg,
            "final",
            canonical_structure,
            queries=resolve(paper["confirmation_candidates"]),
            labels=resolve(paper["confirmation_labels"]),
        ),
        "final_evaluation_freeze": freeze_eval_command(
            cfg, resolve(cfg["evaluation_freeze"]["canonical_output_dir"])
        ),
        "reranker_grouped_cv_freeze": reranker_split_command(
            cfg,
            resolve(
                cfg["evaluation_freeze"]["reranker_cv"]["output"]
            ),
        ),
        "materialize_query_structure_dataset": [
            sys.executable,
            str(Path("tools") / "freeze_research_data_partitions.py"),
            "materialize-query-structure-dataset",
            "--out-dir",
            str(resolve(
                cfg["query_structure_dataset_release"]["output_dir"]
            )),
        ],
        "portable_partition_rebuild": [
            sys.executable,
            str(Path("tools") / "freeze_research_data_partitions.py"),
            "rebuild-open-source-partitions",
            "--bundle-root",
            "/path/to/released-partition-bundle",
        ],
    }


def _historical_split_command(
    cfg: dict,
    out_dir: Path,
    *,
    original_audit: Path | None = None,
    expanded_audit: Path | None = None,
    exclude_paths: list[Path] | None = None,
) -> list[str]:
    mixed = cfg["graph_build_corpus"]["historical_mixed_split"]
    command = [
        sys.executable, str(resolve(mixed["script"])),
        "--original-audit",
        str(original_audit or resolve(mixed["original_audit"])),
        "--expanded-audit",
        str(expanded_audit or resolve(mixed["expanded_audit"])),
        "--validation-n", str(mixed["validation_n"]),
        "--test-n", str(mixed["test_n"]),
        "--human-validation-n", str(mixed["human_validation_n"]),
        "--human-test-n", str(mixed["human_test_n"]),
        "--balance", str(mixed["balance"]),
        "--seed", str(mixed["seed"]),
        "--out-dir", str(out_dir),
    ]
    exclusions = (
        exclude_paths
        if exclude_paths is not None
        else [resolve(mixed["exclude"])]
    )
    if exclusions:
        insert_at = 6
        command[insert_at:insert_at] = [
            "--exclude",
            *(str(path) for path in exclusions),
        ]
    return command


def _run(command: list[str]) -> None:
    env = dict(os.environ)
    pythonpath = str(PROJECT_ROOT)
    env["PYTHONPATH"] = (
        pythonpath
        if not env.get("PYTHONPATH")
        else pythonpath + os.pathsep + env["PYTHONPATH"]
    )
    try:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "rebuild command failed\n"
            f"command={command!r}\nstdout={error.stdout}\nstderr={error.stderr}"
        ) from error


def _run_graph(
    command: list[str],
    *,
    backend: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    env = dict(os.environ)
    roots = [str(PROJECT_ROOT)]
    if legacy_root := os.environ.get("EVIDENCE_PIPELINE_LEGACY_GRAPH_ROOT"):
        roots.append(legacy_root)
    if env.get("PYTHONPATH"):
        roots.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(roots)
    if backend:
        env["STUDY_RETR_BACKEND"] = backend
    if extra_env:
        env.update(extra_env)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    try:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "graph rebuild command failed\n"
            f"command={command!r}\nstdout={error.stdout}\nstderr={error.stderr}"
        ) from error


def _pair_set_sha256(path: Path, fields: tuple[str, ...]) -> tuple[int, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        pairs = {
            tuple(str(row.get(field) or "").strip() for field in fields)
            for row in csv.DictReader(handle)
        }
    pairs.discard(tuple("" for _ in fields))
    payload = "".join(
        "\t".join(pair) + "\n" for pair in sorted(pairs)
    ).encode("utf-8")
    return len(pairs), hashlib.sha256(payload).hexdigest()


def rebuild_construction(cfg: dict, out_dir: Path) -> dict:
    corpus = cfg["graph_build_corpus"]
    mixed_dir = out_dir / "mixed_query_splits_800"
    cap_dir = out_dir / "expanded_graph_rebuild"
    unified_path = out_dir / "unified_extract_input_1618.csv"
    mixed_dir.mkdir(parents=True, exist_ok=True)
    cap_dir.mkdir(parents=True, exist_ok=True)

    _run(_historical_split_command(cfg, mixed_dir))
    split_comparison = {}
    canonical_mixed = resolve(
        corpus["historical_mixed_split"]["output_dir"]
    )
    for name in (
        "validation_queries.csv",
        "test_queries.csv",
        "reserve_queries.csv",
        "all_mixed_query_candidates.csv",
    ):
        rebuilt_hash = sha256(mixed_dir / name)
        canonical_hash = sha256(canonical_mixed / name)
        split_comparison[name] = {
            "rebuilt_sha256": rebuilt_hash,
            "canonical_sha256": canonical_hash,
            "byte_identical": rebuilt_hash == canonical_hash,
        }
    if not all(item["byte_identical"] for item in split_comparison.values()):
        raise ValueError("historical construction split is not byte reproducible")

    _run([
        sys.executable, str(resolve(corpus["cap_script"])),
        "--input", str(resolve(corpus["source"])),
        "--out-dir", str(cap_dir),
        "--exclude-post-jsonl", str(resolve(corpus["exclude_posts"])),
        "--shallow-cap", str(corpus["shallow_cap"]),
        "--mid-cap", str(corpus["mid_cap"]),
        "--deep-cap", str(corpus["deep_cap"]),
    ])
    cap_comparison = {}
    canonical_cap = resolve(corpus["cap_output_dir"])
    for name in (
        "expanded_comment_input_cap30.csv",
        "expanded_post_input_cap30.csv",
        "post_problem_annotations_extra.csv",
        "post_texts_extra.csv",
    ):
        rebuilt_hash = sha256(cap_dir / name)
        canonical_hash = sha256(canonical_cap / name)
        cap_comparison[name] = {
            "rebuilt_sha256": rebuilt_hash,
            "canonical_sha256": canonical_hash,
            "byte_identical": rebuilt_hash == canonical_hash,
        }
    if not all(item["byte_identical"] for item in cap_comparison.values()):
        raise ValueError("cap30 graph corpus is not byte reproducible")

    unified = corpus["unified_extraction_selection"]
    _run([
        sys.executable, str(resolve(unified["script"])),
        "--old-comments", str(resolve(unified["old_comments"])),
        "--expanded-cap30", str(cap_dir / "expanded_comment_input_cap30.csv"),
        "--val", str(mixed_dir / "validation_queries.csv"),
        "--test", str(mixed_dir / "test_queries.csv"),
        "--seed", str(unified["seed"]),
        "--out", str(unified_path),
    ])
    canonical_unified = resolve(unified["output"])
    rebuilt_pairs = _pair_set_sha256(
        unified_path, ("post_id", "comment_id")
    )
    canonical_pairs = _pair_set_sha256(
        canonical_unified, ("post_id", "comment_id")
    )
    if rebuilt_pairs != canonical_pairs:
        raise ValueError("unified extraction row identities differ from canonical")

    return {
        "status": "passed",
        "historical_construction_split": split_comparison,
        "cap30_graph_corpus": cap_comparison,
        "unified_extraction_selection": {
            "row_identity_count": rebuilt_pairs[0],
            "row_identity_set_sha256": rebuilt_pairs[1],
            "row_identities_equal": True,
            "byte_identical": sha256(unified_path) == sha256(canonical_unified),
            "byte_note": (
                "Legacy output order depended on Python set iteration. The "
                "current script sorts IDs before the seeded shuffle; membership "
                "is identical and future rebuilds are byte deterministic."
            ),
        },
    }


def rebuild_raw_graph_sample(cfg: dict, out_dir: Path) -> dict:
    output = out_dir / "resample_input_raw.csv"
    manifest = out_dir / "resample_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_graph(_raw_graph_command(cfg, output, manifest))
    canonical_spec = cfg["locked_artifacts"]["expanded_graph_sample"]
    rows, ids = read_ids(output, str(canonical_spec["id_field"]))
    rebuilt_hash = sha256(output)
    result = {
        "status": "passed",
        "rows": rows,
        "unique_ids": len(ids),
        "id_set_sha256": id_set_sha256(ids),
        "sha256": rebuilt_hash,
        "canonical_sha256": canonical_spec["sha256"],
        "byte_identical": rebuilt_hash == canonical_spec["sha256"],
    }
    if (
        rows != canonical_spec["rows"]
        or len(ids) != canonical_spec["unique_ids"]
        or result["id_set_sha256"] != canonical_spec["id_set_sha256"]
        or not result["byte_identical"]
    ):
        raise ValueError(f"raw graph sample rebuild differs: {result}")
    return result


def _dataframe_set_sha256(path: Path) -> tuple[int, str]:
    import pandas as pd

    frame = pd.read_parquet(path).fillna("")
    columns = sorted(str(column) for column in frame.columns)
    rows = {
        tuple(str(row[column]) for column in columns)
        for _, row in frame.iterrows()
    }
    payload = json.dumps(
        {"columns": columns, "rows": sorted(rows)},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(rows), hashlib.sha256(payload).hexdigest()


def _csv_row_set_sha256(path: Path) -> tuple[int, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = sorted(reader.fieldnames or [])
        rows = {
            tuple(str(row.get(column) or "") for column in columns)
            for row in reader
        }
    payload = json.dumps(
        {"columns": columns, "rows": sorted(rows)},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(rows), hashlib.sha256(payload).hexdigest()


def _dense_edge_identities(path: Path) -> tuple[set[tuple[str, ...]], dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    identities = {
        (
            str(row["source"]),
            str(row["target"]),
            str(row["kind"]),
            str(row["relation"]),
        )
        for row in rows
    }
    counts: dict[str, int] = {}
    for row in rows:
        kind = str(row["kind"])
        counts[kind] = counts.get(kind, 0) + 1
    return identities, dict(sorted(counts.items()))


def rebuild_graph_assembly(cfg: dict, out_dir: Path) -> dict:
    assembly = cfg["graph_build_corpus"]["graph_assembly"]
    graph_dir = out_dir / "graph_expanded_t072"
    communities = graph_dir / "communities.json"
    dense_edges = graph_dir / "dense_edges.csv"
    candidate_dense_edges = graph_dir / "dense_edges_recomputed.csv"
    graph_dir.mkdir(parents=True, exist_ok=True)

    _run_graph([
        sys.executable,
        str(resolve(assembly["build_script"])),
        "--remapped-dir", str(resolve(assembly["remapped_dir"])),
        "--compiled-dir", str(resolve(assembly["compiled_dir"])),
        "--entities-dir", str(resolve(assembly["entities_dir"])),
        "--entity-jsonl", str(resolve(assembly["entity_jsonl"])),
        "--out-dir", str(graph_dir),
    ])
    canonical_graph = resolve(assembly["output_dir"])
    table_checks = {}
    for name in ("graph_nodes.parquet", "graph_edges.parquet"):
        rebuilt = _dataframe_set_sha256(graph_dir / name)
        canonical = _dataframe_set_sha256(canonical_graph / name)
        table_checks[name] = {
            "row_identity_count": rebuilt[0],
            "row_identity_set_sha256": rebuilt[1],
            "canonical_set_sha256": canonical[1],
            "semantic_rows_equal": rebuilt == canonical,
            "byte_identical": sha256(graph_dir / name)
            == sha256(canonical_graph / name),
        }
        if rebuilt != canonical:
            raise ValueError(f"{name} semantic rows differ from canonical graph")

    # The builder directly writes Parquet. Materialise full CSVs for the
    # historical densifier and for the isolated Python<=3.12 Leiden runtime.
    import pandas as pd
    for stem in ("graph_nodes", "graph_edges"):
        pd.read_parquet(graph_dir / f"{stem}.parquet").to_csv(
            graph_dir / f"{stem}.csv", index=False
        )

    _run_graph([
        sys.executable,
        str(resolve(assembly["densify_script"])),
        "--graph", str(graph_dir),
        "--cooc-min", str(assembly["cooc_min"]),
        "--knn-k", str(assembly["knn_k"]),
        "--knn-thresh", str(assembly["knn_thresh"]),
        "--normalize", str(assembly["normalize"]),
        "--relation-weight", str(assembly["relation_weight"]),
        "--cooc-weight", str(assembly["cooc_weight"]),
        "--knn-weight", str(assembly["knn_weight"]),
        "--max-cluster-size", str(assembly["max_cluster_size"]),
        "--seed", str(assembly["leiden_seed"]),
        "--edges-only",
        "--edge-out", str(candidate_dense_edges),
    ], backend=str(assembly["semantic_backend"]), extra_env={
        "STUDY_BERT_MODEL": str(assembly["semantic_model"]),
        "STUDY_BERT_REVISION": str(assembly["semantic_model_revision"]),
        "STUDY_BERT_BATCH_SIZE": str(
            assembly["semantic_embedding_batch_size"]
        ),
        "STUDY_KNN_IMPLEMENTATION": str(assembly["knn_implementation"]),
    })
    canonical_dense_edges = resolve(assembly["dense_edges"])
    recomputed_ids, recomputed_counts = _dense_edge_identities(
        candidate_dense_edges
    )
    canonical_ids, canonical_counts = _dense_edge_identities(
        canonical_dense_edges
    )
    non_knn_recomputed = {
        row for row in recomputed_ids if row[2] != "knn"
    }
    non_knn_canonical = {
        row for row in canonical_ids if row[2] != "knn"
    }
    if non_knn_recomputed != non_knn_canonical:
        raise ValueError("relation/co-occurrence edge identities differ from canonical")

    leiden_python = os.environ.get(
        str(assembly["community_python_env"]), sys.executable
    )
    _run_graph([
        leiden_python,
        str(resolve(assembly["densify_script"])),
        "--graph", str(graph_dir),
        "--cooc-min", str(assembly["cooc_min"]),
        "--knn-k", str(assembly["knn_k"]),
        "--knn-thresh", str(assembly["knn_thresh"]),
        "--normalize", str(assembly["normalize"]),
        "--relation-weight", str(assembly["relation_weight"]),
        "--cooc-weight", str(assembly["cooc_weight"]),
        "--knn-weight", str(assembly["knn_weight"]),
        "--max-cluster-size", str(assembly["max_cluster_size"]),
        "--seed", str(assembly["leiden_seed"]),
        "--dense-edge-input", str(canonical_dense_edges),
        "--out", str(communities),
        "--edge-out", str(dense_edges),
    ], backend=str(assembly["semantic_backend"]))
    rebuilt_communities = json.loads(communities.read_text(encoding="utf-8"))
    canonical_communities = json.loads(
        resolve(assembly["communities"]).read_text(encoding="utf-8")
    )
    rebuilt_mapping = rebuilt_communities["entity_community"]
    canonical_mapping = canonical_communities["entity_community"]
    communities_equal = rebuilt_mapping == canonical_mapping
    if not communities_equal:
        raise ValueError("Leiden entity-community mapping differs from canonical")
    dense_edge_rows = _csv_row_set_sha256(dense_edges)
    canonical_dense_edge_rows = _csv_row_set_sha256(
        canonical_dense_edges
    )
    if dense_edge_rows != canonical_dense_edge_rows:
        raise ValueError("densified edge rows differ from canonical")
    graph_summary_equal = json.loads(
        (graph_dir / "graph_summary.json").read_text(encoding="utf-8")
    ) == json.loads(
        (canonical_graph / "graph_summary.json").read_text(encoding="utf-8")
    )
    if not graph_summary_equal:
        raise ValueError("graph summary differs from canonical")
    community_metadata_keys = ("edge_counts", "n_communities", "backend", "weighting")
    communities_semantically_equal = all(
        rebuilt_communities[key] == canonical_communities[key]
        for key in community_metadata_keys
    )
    if not communities_semantically_equal:
        raise ValueError("community method metadata differs from canonical")
    return {
        "status": "passed",
        "graph_tables": table_checks,
        "graph_summary_equal": graph_summary_equal,
        "dense_edges": {
            "row_identity_count": dense_edge_rows[0],
            "row_identity_set_sha256": dense_edge_rows[1],
            "canonical_set_sha256": canonical_dense_edge_rows[1],
            "semantic_rows_equal": True,
            "byte_identical": sha256(dense_edges)
            == sha256(canonical_dense_edges),
        },
        "densification_recomputation_audit": {
            "runtime_boundary": assembly["canonical_dense_edge_boundary"],
            "recomputed_counts": recomputed_counts,
            "canonical_counts": canonical_counts,
            "identity_overlap": len(recomputed_ids & canonical_ids),
            "recomputed_only": len(recomputed_ids - canonical_ids),
            "canonical_only": len(canonical_ids - recomputed_ids),
            "all_identities_equal": recomputed_ids == canonical_ids,
            "byte_identical": sha256(candidate_dense_edges)
            == sha256(canonical_dense_edges),
            "non_knn_identities_equal": True,
        },
        "community_assignment_count": len(rebuilt_mapping),
        "community_assignments_equal": communities_equal,
        "community_metadata_equal": communities_semantically_equal,
        "community_file_byte_identical": sha256(communities)
        == sha256(resolve(assembly["communities"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)

    rebuild_parser = sub.add_parser("rebuild-evaluation")
    rebuild_parser.add_argument("--out-dir", type=Path)

    construction_parser = sub.add_parser("rebuild-construction")
    construction_parser.add_argument("--out-dir", type=Path)

    graph_parser = sub.add_parser("rebuild-graph-assembly")
    graph_parser.add_argument("--out-dir", type=Path)

    raw_parser = sub.add_parser("rebuild-raw-graph-sample")
    raw_parser.add_argument("--out-dir", type=Path)

    paper_parser = sub.add_parser("rebuild-paper-evaluation-cohorts")
    paper_parser.add_argument("--out-dir", type=Path)

    portable_parser = sub.add_parser("rebuild-open-source-partitions")
    portable_parser.add_argument("--out-dir", type=Path)
    portable_parser.add_argument(
        "--bundle-root",
        type=Path,
        help=(
            "Root of the released six-input partition bundle. Omit this "
            "option to verify the same chain against canonical local inputs."
        ),
    )

    export_parser = sub.add_parser("export-open-source-partition-bundle")
    export_parser.add_argument("--out-dir", type=Path, required=True)
    export_parser.add_argument(
        "--acknowledge-sensitive-content",
        action="store_true",
        help=(
            "Required acknowledgement that the local bundle contains Reddit "
            "post text and still needs ethics/licence/privacy review."
        ),
    )

    dataset_parser = sub.add_parser(
        "materialize-query-structure-dataset"
    )
    dataset_parser.add_argument("--out-dir", type=Path)

    development300_parser = sub.add_parser("freeze-development300")
    development300_parser.add_argument("--out-dir", type=Path)

    sub.add_parser("print-open-source-bundle")
    sub.add_parser("print-commands")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.command == "verify":
        result = verify(cfg, args.config)
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps({
            "status": result["status"],
            "artifacts": len(result["artifacts"]),
            "invariants": len(result["invariants"]),
            "manifest": str(args.manifest_out),
        }, ensure_ascii=False, indent=2))
    elif args.command == "rebuild-evaluation":
        if args.out_dir:
            result = rebuild_evaluation(cfg, args.out_dir)
        else:
            with tempfile.TemporaryDirectory(prefix="partition-rebuild-") as tmp:
                result = rebuild_evaluation(cfg, Path(tmp))
                result["out_dir"] = "temporary_directory_removed_after_verification"
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "rebuild-construction":
        if args.out_dir:
            result = rebuild_construction(cfg, args.out_dir)
        else:
            with tempfile.TemporaryDirectory(
                prefix="construction-rebuild-"
            ) as tmp:
                result = rebuild_construction(cfg, Path(tmp))
                result["out_dir"] = (
                    "temporary_directory_removed_after_verification"
                )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "rebuild-graph-assembly":
        if args.out_dir:
            result = rebuild_graph_assembly(cfg, args.out_dir)
        else:
            with tempfile.TemporaryDirectory(prefix="graph-rebuild-") as tmp:
                result = rebuild_graph_assembly(cfg, Path(tmp))
                result["out_dir"] = (
                    "temporary_directory_removed_after_verification"
                )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "rebuild-raw-graph-sample":
        if args.out_dir:
            result = rebuild_raw_graph_sample(cfg, args.out_dir)
        else:
            with tempfile.TemporaryDirectory(prefix="raw-graph-rebuild-") as tmp:
                result = rebuild_raw_graph_sample(cfg, Path(tmp))
                result["out_dir"] = (
                    "temporary_directory_removed_after_verification"
                )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "rebuild-paper-evaluation-cohorts":
        if args.out_dir:
            result = rebuild_paper_evaluation_cohorts(cfg, args.out_dir)
        else:
            with tempfile.TemporaryDirectory(
                prefix="paper-cohort-rebuild-"
            ) as tmp:
                result = rebuild_paper_evaluation_cohorts(cfg, Path(tmp))
                result["out_dir"] = (
                    "temporary_directory_removed_after_verification"
                )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "rebuild-open-source-partitions":
        bundle_root = args.bundle_root.resolve() if args.bundle_root else None
        if args.out_dir:
            result = rebuild_open_source_partitions(
                cfg,
                args.out_dir,
                bundle_root=bundle_root,
            )
        else:
            with tempfile.TemporaryDirectory(
                prefix="portable-partition-rebuild-"
            ) as tmp:
                result = rebuild_open_source_partitions(
                    cfg,
                    Path(tmp),
                    bundle_root=bundle_root,
                )
                result["out_dir"] = (
                    "temporary_directory_removed_after_verification"
                )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "print-open-source-bundle":
        print(json.dumps(
            open_source_bundle_inventory(cfg),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ))
    elif args.command == "export-open-source-partition-bundle":
        if not args.acknowledge_sensitive_content:
            parser.error(
                "export requires --acknowledge-sensitive-content because the "
                "bundle contains Reddit post text"
            )
        result = export_open_source_partition_bundle(cfg, args.out_dir)
        print(json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ))
    elif args.command == "materialize-query-structure-dataset":
        out_dir = (
            args.out_dir
            if args.out_dir
            else resolve(
                cfg["query_structure_dataset_release"]["output_dir"]
            )
        )
        result = materialize_query_structure_dataset(cfg, out_dir)
        print(json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ))
    elif args.command == "freeze-development300":
        out_dir = (
            args.out_dir
            if args.out_dir
            else resolve(
                cfg["paper_evaluation_cohorts"]["development300"]["output_dir"]
            )
        )
        result = freeze_development300(cfg, out_dir)
        print(json.dumps({
            "status": result["status"],
            "counts": result["counts"],
            "label_counts": result["label_counts"],
            "out_dir": str(out_dir),
        }, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(
            reproducibility_commands(cfg), ensure_ascii=False, indent=2
        ))


if __name__ == "__main__":
    main()
