# 【文件 090】核对效用标签字段、身份和覆盖完整性，拒绝把缺标签当负例
# 【流程位置】检索、效用与社区对应评价；所属包：evaluation
# 【主要函数】CompletenessResult, assess_utility_v2_judgment, complete_utility_v2_rows
# 【输入接口】rows 等函数参数；返回值及写出操作见对应函数
# 【调用方】candidate_pool/analyze_strict_sbert_graph_oracle.py
# 【调用方】candidate_pool/run_dense_semantic_drift_rescue_audit.py
# 【调用方】candidate_pool/run_m50_dense_frontier_analysis.py
# 【调用方】candidate_pool/run_m50_graph_frontier_analysis.py
# 【调用方】fusion/run_depth_graph_utility_community_frontier.py
# 【调用方】utility_scoring/annotation/run_coverage_complete_residual_judging.py
# 【调用方】utility_scoring/annotation/run_top3_residual_judging.py
# 【调用方】utility_scoring/build_stage2_redesign_features.py
# 【调用方】utility_scoring/run_stage2_redesign_crossencoder.py
# 【调用方】utility_scoring/stage2_training_contract.py
# 【调用方】evidence_selection/run_selection_action_space_repair.py
# 【调用方】evidence_selection/run_set_aware_selection_ablation.py
# 【调用方】evidence_selection/run_strict_native_graph_conservative_policy.py
# 【调用方】evidence_selection/run_strict_sbert_mixed_selector.py
# 【调用方】evaluation/analyze_rq2b_set_correspondence.py
# 【调用方】evaluation/community_reply_auxiliary.py
# 【调用方】evaluation/confirmatory_test200_rq2b.py
# 【调用方】evaluation/fusion_strategy_ablation.py
# 【调用方】evaluation/run_evidence_signal_triangulation.py
# 【调用方】evaluation/run_m50_community_frontier_analysis.py

"""Canonical completeness checks for utility-v2 judgment registries.

The legacy ``judgment_status`` field describes candidate-pool state and is not
authoritative after a judgment has been parsed.  Completion is established by
the parsed utility, all six validated dimensions, judge provenance, exact pair
identity, and an explicit or inferable successful parse/validation state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Any


DIMS_V2 = ("relevance", "usefulness", "novelty", "actionability", "resonance", "safety")


@dataclass(frozen=True)
class CompletenessResult:
    complete: bool
    reasons: tuple[str, ...]
    validation_basis: str | None


def _finite_number(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return parsed == parsed and parsed not in (float("inf"), float("-inf"))


def assess_utility_v2_judgment(row: Mapping[str, Any]) -> CompletenessResult:
    """Assess one row without consulting the non-authoritative status string."""
    reasons: list[str] = []
    if not str(row.get("query_id") or "").strip():
        reasons.append("missing_query_id")
    if not str(row.get("comment_id") or "").strip():
        reasons.append("missing_comment_id")
    if not _finite_number(row.get("utility")):
        reasons.append("invalid_utility")

    for dim in DIMS_V2:
        value = row.get(f"label_{dim}")
        if not _finite_number(value) or not 1 <= float(value) <= 7:
            reasons.append(f"invalid_label_{dim}")

    has_model = bool(str(row.get("judge_model") or row.get("model") or "").strip())
    has_protocol = bool(str(
        row.get("judge_id") or row.get("judge_version") or row.get("prompt_sha256") or ""
    ).strip())
    if not has_model:
        reasons.append("missing_judge_model")
    if not has_protocol:
        reasons.append("missing_judge_protocol_provenance")

    status = row.get("validation_status")
    if status is not None and str(status).lower() not in {"valid", "validated", "pass", "passed"}:
        reasons.append("explicit_validation_failed")
        basis = None
    elif status is not None:
        basis = "explicit_validation_status"
    else:
        # Historical utility-v2 rows predate an explicit validation_status.
        # Successfully materialised, range-valid parsed fields are the legacy
        # equivalent of a passed parser/validator.
        basis = "legacy_parsed_fields_inferred_valid"

    return CompletenessResult(not reasons, tuple(reasons), basis if not reasons else None)


def complete_utility_v2_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    """Return complete rows and an exact pair registry; reject duplicates."""
    complete = [dict(row) for row in rows if assess_utility_v2_judgment(row).complete]
    registry = {(str(row["query_id"]), str(row["comment_id"])): row for row in complete}
    if len(registry) != len(complete):
        raise ValueError("duplicate completed utility-v2 judgment pair")
    return complete, registry

