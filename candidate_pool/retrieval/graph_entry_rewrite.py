"""Frozen-summary graph-entry rewrites with deterministic hard validation.

This module extends the canonical rewrite lifecycle in ``query_rewrite``.  It
does not retrieve, read labels, inspect routes, or modify the graph.  The
original query is authoritative; the frozen summary supplies stable IDs only.
"""
from __future__ import annotations

import json
import re

from candidate_pool.retrieval.query_rewrite import (
    BaseQueryRewriter,
    QueryRewriteParameters,
    RewriteResult,
    content_tokens,
)


REQUIRED_FIELDS = (
    "primary_rewrite", "supporting_rewrites", "preserved_need_ids",
    "preserved_constraint_ids", "source_spans", "introduced_content",
    "uncertainties",
)
GENERIC_PATTERNS = (
    r"^adhd advice$", r"^adhd help$", r"^adhd tips$",
    r"^productivity advice$", r"^mental health advice$",
)
UNCERTAINTY_MARKERS = ("maybe", "might", "possibly", "unsure", "not sure",
                       "i think", "i feel", "seems", "could be")


def _span_normal_form(text: str) -> str:
    """Normalize only one-codepoint typography for deterministic alignment."""
    return str(text).translate(str.maketrans({
        "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-",
        "\u00a0": " ",
    }))


def correct_source_spans(payload: dict, query_text: str) -> int:
    """Correct metadata offsets/text without changing generated rewrites.

    Exact unique matches are relocated. Typographic-only differences are
    aligned against a length-preserving normal form, after which the exact
    original substring replaces the model's normalized copy. Ambiguous or
    substantive mismatches remain untouched and fail normal validation.
    """
    corrected = 0
    normalized_query = _span_normal_form(query_text)
    for span in payload.get("source_spans") or []:
        if not isinstance(span, dict):
            continue
        text = str(span.get("text") or "")
        start, end = span.get("start_char"), span.get("end_char")
        if isinstance(start, int) and isinstance(end, int) and \
                0 <= start <= end <= len(query_text) and query_text[start:end] == text:
            continue
        exact_positions = [m.start() for m in re.finditer(re.escape(text), query_text)] \
            if text else []
        if len(exact_positions) == 1:
            new_start = exact_positions[0]
            span.update(start_char=new_start, end_char=new_start + len(text))
            corrected += 1
            continue
        normalized = _span_normal_form(text)
        normalized_positions = [m.start() for m in re.finditer(
            re.escape(normalized), normalized_query)] if normalized else []
        if len(normalized_positions) == 1:
            new_start = normalized_positions[0]
            new_end = new_start + len(normalized)
            # The normal form is deliberately length preserving.
            span.update(text=query_text[new_start:new_end],
                        start_char=new_start, end_char=new_end)
            corrected += 1
    return corrected


class FrozenSummaryGraphEntryRewriter(BaseQueryRewriter):
    """One strict generator used for compression or structured conditions."""

    required_fields = REQUIRED_FIELDS

    def __init__(self, params: QueryRewriteParameters, *, mode: str,
                 generate_fn=None, cache=None):
        if mode not in {"compression", "structured"}:
            raise ValueError(f"unknown graph-entry rewrite mode: {mode}")
        self.mode = mode
        self.method_name = f"graph_entry_{mode}"
        self.prompt_name = (
            "query_rewrite_graph_entry_compression_v2" if mode == "compression"
            else "query_rewrite_graph_entry_structured_v1")
        super().__init__(params, generate_fn=generate_fn, cache=cache)
        self._active_summary: dict = {}

    def build_prompt_with_summary(self, query_text: str, summary: dict) -> str:
        return self._template.replace(
            "{QUERY_TEXT}", str(query_text).strip()).replace(
            "{QUERY_SUMMARY_JSON}", json.dumps(
                summary, ensure_ascii=False, sort_keys=True))

    def build_prompt(self, query_text: str) -> str:
        return self.build_prompt_with_summary(query_text, self._active_summary)

    def extract_rewrites(self, payload: dict) -> tuple[str, list[str]]:
        supporting = payload.get("supporting_rewrites") or []
        return payload.get("primary_rewrite"), [
            str(item.get("text") or "") for item in supporting
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]

    @staticmethod
    def _summary_ids(summary: dict) -> tuple[set[str], set[str]]:
        needs = set()
        primary = summary.get("primary_need") or {}
        if primary.get("need_id"):
            needs.add(str(primary["need_id"]))
        needs.update(str(row["need_id"]) for row in
                     (summary.get("additional_needs") or [])
                     if isinstance(row, dict) and row.get("need_id"))
        constraints = {str(row["constraint_id"]) for row in
                       (summary.get("constraints") or [])
                       if isinstance(row, dict) and row.get("constraint_id")}
        return needs, constraints

    def method_validate(self, payload: dict, query_text: str) -> list[str]:
        flags = []
        corrected = correct_source_spans(payload, query_text)
        if corrected:
            flags.append(f"source_spans_auto_corrected:{corrected}")
        summary = self._active_summary
        need_ids, constraint_ids = self._summary_ids(summary)
        preserved_needs = {str(x) for x in payload.get("preserved_need_ids") or []}
        preserved_constraints = {
            str(x) for x in payload.get("preserved_constraint_ids") or []}
        unknown_needs = preserved_needs - need_ids
        unknown_constraints = preserved_constraints - constraint_ids
        if unknown_needs:
            flags.append("unknown_need_ids:" + ",".join(sorted(unknown_needs)))
        if unknown_constraints:
            flags.append("unknown_constraint_ids:" + ",".join(sorted(unknown_constraints)))
        if "N1" in need_ids and "N1" not in preserved_needs:
            flags.append("primary_need_omitted:N1")
        missing_constraints = constraint_ids - preserved_constraints
        if missing_constraints:
            flags.append("constraints_omitted:" + ",".join(sorted(missing_constraints)))

        spans = payload.get("source_spans") or []
        if not spans:
            flags.append("no_source_spans")
        for pos, span in enumerate(spans):
            if not isinstance(span, dict):
                flags.append(f"source_span_not_object:{pos}")
                continue
            text = str(span.get("text") or "")
            start, end = span.get("start_char"), span.get("end_char")
            if not isinstance(start, int) or not isinstance(end, int):
                flags.append(f"source_span_offset_type:{pos}")
            elif start < 0 or end < start or query_text[start:end] != text:
                flags.append(f"source_span_mismatch:{pos}")

        introduced = payload.get("introduced_content") or []
        if introduced:
            flags.append("introduced_content_declared")
        supporting = payload.get("supporting_rewrites") or []
        if not isinstance(supporting, list):
            flags.append("supporting_rewrites_not_list")
            supporting = []
        if self.mode == "compression" and supporting:
            flags.append("compression_has_supporting_rewrites")
        if len(supporting) > 2:
            flags.append("too_many_supporting_rewrites")
        for pos, item in enumerate(supporting):
            if not isinstance(item, dict):
                flags.append(f"supporting_rewrite_not_object:{pos}")
                continue
            expected_id = f"R{pos + 2}"
            if item.get("rewrite_id") != expected_id:
                flags.append(f"supporting_rewrite_id_invalid:{pos}")
            linked_needs = {str(x) for x in item.get("linked_need_ids") or []}
            linked_constraints = {
                str(x) for x in item.get("linked_constraint_ids") or []}
            if not linked_needs and not linked_constraints:
                flags.append(f"supporting_rewrite_unlinked:{pos}")
            if linked_needs - need_ids:
                flags.append(f"supporting_unknown_need:{pos}")
            if linked_constraints - constraint_ids:
                flags.append(f"supporting_unknown_constraint:{pos}")

        combined = " ".join([str(payload.get("primary_rewrite") or ""), *[
            str(x.get("text") or "") for x in supporting if isinstance(x, dict)]])
        if any(re.fullmatch(pattern, combined.strip(), flags=re.I)
               for pattern in GENERIC_PATTERNS):
            flags.append("overly_generic_rewrite")
        # Flag certainty strengthening when the source contains uncertainty
        # but the rewrite drops every such marker. This proxy is conservative.
        lower_original, lower_rewrite = query_text.lower(), combined.lower()
        if any(m in lower_original for m in UNCERTAINTY_MARKERS) and not any(
                m in lower_rewrite for m in UNCERTAINTY_MARKERS):
            flags.append("possible_uncertainty_strengthening")
        rewrite_tokens = content_tokens(combined)
        if rewrite_tokens and (
            len(rewrite_tokens - content_tokens(query_text)) / len(rewrite_tokens)
            > 0.35
        ):
            flags.append("high_novel_token_fraction_strict")
        return flags

    def rewrite_with_summary(self, query_id: str, query_text: str,
                             summary: dict) -> RewriteResult:
        self._active_summary = summary
        result = super().rewrite(query_id, query_text)
        fatal_prefixes = (
            "unknown_need_ids:", "unknown_constraint_ids:",
            "primary_need_omitted:", "constraints_omitted:",
            "no_source_spans", "source_span_not_object:",
            "source_span_offset_type:", "source_span_mismatch:",
            "introduced_content_declared", "supporting_rewrites_not_list",
            "compression_has_supporting_rewrites", "too_many_supporting_rewrites",
            "supporting_rewrite_not_object:", "supporting_rewrite_id_invalid:",
            "supporting_rewrite_unlinked:", "supporting_unknown_need:",
            "supporting_unknown_constraint:", "overly_generic_rewrite",
        )
        fatal = [flag for flag in result.validation_flags
                 if flag.startswith(fatal_prefixes)]
        if result.status == "ok" and fatal:
            result.status = "failed"
            result.failure_reason = f"validation:{fatal[0]}"
            result.rewrite_text = ""
            result.subqueries = []
        return result
