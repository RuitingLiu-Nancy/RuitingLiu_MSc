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

