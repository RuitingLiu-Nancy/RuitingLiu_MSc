"""Round-3 structured query rewrite pipeline (docs_v2/48 Prompt 1).

Four faithfulness-constrained rewriters over the SAME frozen Official
HippoRAG2 graph/index (no graph writes, no OpenIE reruns, no label reads):

- ``FaithfulCompressionRewriter``   — adapted from the query-rewriting-for-RAG
  line (Rewrite-Retrieve-Read, Ma et al. 2023): frozen zero-shot prompt, no
  trained rewriter, with explicit preserve/remove contract.
- ``NeedConstraintDecompositionRewriter`` — extends this project's Section17
  five-slot decomposition (``section17_query_decomposition`` prompt) with
  standalone per-need retrieval queries carrying their relevant constraints.
- ``ConceptStepBackRewriter``       — adapted Step-Back Prompting (Zheng et
  al., ICLR 2024): abstraction is constrained to neutral concepts, each
  grounded in an exact source span of the post.
- ``ExpectedEvidenceRewriter``      — adapted HyDE (Gao et al. 2023): instead
  of a hypothetical *answer* document, describes required properties of the
  sought evidence, prohibited from inventing solutions.

Shared machinery (docs_v2/48 §2.1): deterministic config, content-addressed
generation cache (reuses ``iterative_retrieval.GenerationCache``), retry,
JSON-schema validation with one malformed-JSON repair attempt, prompt
versioning (name + sha256), per-call char/estimated-token logging, and
route-blind safety checks.  A single query failure never blocks the batch:
the ``original`` channel is always preserved and the failure reason archived.

Provenance ledger: docs_v2/11 §F.  All tunables live in
``configuration/params.yaml::query_rewrite_pipeline``.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from candidate_pool.retrieval.iterative_retrieval import GenerationCache


REWRITE_METHODS = (
    "compression", "decomposition", "concept_step_back", "expected_evidence")

# Unified metadata view (docs_v2/48 §2.1): every method fills its own fields,
# the rest stay null so downstream readers see one stable schema.
UNIFIED_METADATA_FIELDS = (
    "primary_need", "additional_needs", "constraints", "failed_attempts",
    "desired_outcome", "compressed_query", "subqueries", "concept_query",
    "expected_evidence_query", "ambiguities", "validation_flags")

# Unified per-channel JSONL contract (docs_v2/48 §2.0).  Retrieval-dependent
# fields are null at the Prompt-1 stage and filled by later prompts.
CHANNEL_ROW_FIELDS = (
    "query_id", "original_query", "rewrite_method", "rewrite_text",
    "rewrite_metadata", "fact_hits", "recognised_facts", "seed_nodes",
    "graph_entry_status", "retrieved_comments", "diagnostics")

_WORD = re.compile(r"[a-z][a-z']+")
_STOPWORDS = frozenset("""
a about after all also am an and any are as at be because been before being
but by can cannot could did do does doing down for from had has have having
he her here hers him his how i if in into is it its just like me more most my
myself no nor not now of off on once only or other our out over own same she
so some such than that the their them then there these they this those to too
under until up very was we were what when where which while who whom why will
with would you your yours
""".split())


def content_tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(str(text).lower())
            if len(w) >= 4 and w not in _STOPWORDS}


def extract_json_object(text: str) -> dict:
    """Parse the first top-level JSON object in a generation.

    Recovers from surrounding prose / markdown fences; raises ``ValueError``
    when no parseable object exists (callers archive, never crash the batch).
    """
    text = str(text).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in generation")
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                payload = json.loads(text[start:i + 1])
                if not isinstance(payload, dict):
                    raise ValueError("top-level JSON is not an object")
                return payload
    raise ValueError("unbalanced JSON object in generation")


@dataclass(frozen=True)
class QueryRewriteParameters:
    """All values are read from configuration/params.yaml::query_rewrite_pipeline."""
    llm_model: str = "bedrock:global.anthropic.claude-haiku-4-5-20251001-v1:0"
    llm_max_tokens: int = 1200
    llm_temperature: float = 0.0
    llm_retries: int = 3
    llm_retry_sleep_seconds: float = 5.0
    max_subqueries: int = 6
    max_concepts: int = 10
    # Safety heuristics (route-blind, documented; lists come from params.yaml).
    advice_patterns: tuple[str, ...] = (
        r"\byou should\b", r"\byou could try\b", r"\byou can try\b",
        r"\bi recommend\b", r"\bmy advice\b", r"\btry using\b",
        r"\bworks for me\b", r"\bthe answer is\b", r"\bwhat helped me\b")
    clinical_terms: tuple[str, ...] = (
        "adderall", "ritalin", "vyvanse", "concerta", "elvanse", "strattera",
        "methylphenidate", "amphetamine", "dexamfetamine", "lisdexamfetamine",
        "atomoxetine", "guanfacine", "bupropion", "wellbutrin", "modafinil",
        "autism", "autistic", "bipolar", "borderline", "ocd", "ptsd",
        "depression", "insomnia")
    # Max fraction of rewrite content tokens absent from the original before
    # the high_novel_token_fraction flag fires (abstractive methods higher).
    novel_token_thresholds: dict = field(default_factory=lambda: {
        "compression": 0.35, "decomposition": 0.35,
        "concept_step_back": 0.80, "expected_evidence": 0.80})

    @classmethod
    def from_params(cls, raw: dict | None) -> "QueryRewriteParameters":
        raw = dict(raw or {})
        thresholds = dict(cls().novel_token_thresholds)
        thresholds.update(raw.pop("novel_token_thresholds", {}) or {})
        for key in ("advice_patterns", "clinical_terms"):
            if key in raw and raw[key] is not None:
                raw[key] = tuple(raw[key])
        known = {f for f in cls.__dataclass_fields__}
        return cls(novel_token_thresholds=thresholds,
                   **{k: v for k, v in raw.items()
                      if k in known and k != "novel_token_thresholds"})


@dataclass
class RewriteResult:
    query_id: str
    original_query: str
    rewrite_method: str
    status: str                    # "ok" | "failed"
    rewrite_text: str = ""
    subqueries: list = field(default_factory=list)
    payload: dict = field(default_factory=dict)
    validation_flags: list = field(default_factory=list)
    failure_reason: str | None = None
    provenance: dict = field(default_factory=dict)

    def unified_metadata(self) -> dict:
        meta = {key: None for key in UNIFIED_METADATA_FIELDS}
        for key in UNIFIED_METADATA_FIELDS:
            if key in self.payload:
                meta[key] = self.payload[key]
        # Per-method constraint lists map onto the unified "constraints" view
        # (compression: preserved_constraints; expected_evidence:
        # required_constraints) so downstream constraint-preservation checks
        # read one field regardless of method.
        if meta["constraints"] is None:
            for alias in ("preserved_constraints", "required_constraints"):
                if self.payload.get(alias) is not None:
                    meta["constraints"] = self.payload[alias]
                    break
        if self.subqueries:
            meta["subqueries"] = list(self.subqueries)
        meta["validation_flags"] = list(self.validation_flags)
        return meta


class BaseQueryRewriter:
    """Shared rewrite lifecycle: prompt -> cached LLM -> JSON -> validate.

    Subclasses define ``method_name``, ``prompt_name``, ``required_fields``
    and ``extract_rewrites(payload)``; everything else (cache, retry, repair,
    safety checks, provenance) is common so the four adapters cannot drift.
    """

    method_name: str = ""
    prompt_name: str = ""
    required_fields: tuple = ()

    def __init__(self, params: QueryRewriteParameters, *,
                 generate_fn=None, cache: GenerationCache | None = None):
        self.params = params
        self._generate_fn = generate_fn
        self.cache = cache
        import configuration as _config
        self._template = _config.prompt(self.prompt_name)
        self._prompt_sha = hashlib.sha256(
            self._template.encode("utf-8")).hexdigest()

    # -- prompt / llm ------------------------------------------------------
    def build_prompt(self, query_text: str) -> str:
        return self._template.replace("{QUERY_TEXT}", str(query_text).strip())

    def _generate(self, prompt: str) -> str:
        if self._generate_fn is not None:
            return self._generate_fn(prompt)
        from shared.llm_client import call_chat
        return call_chat(prompt, self.params.llm_model,
                         max_tokens=self.params.llm_max_tokens,
                         temperature=self.params.llm_temperature)

    def _generate_cached(self, prompt: str) -> tuple[str, bool, str | None]:
        if self.cache is not None:
            cached = self.cache.get(prompt)
            if cached is not None:
                return cached, True, None
        error = None
        for attempt in range(self.params.llm_retries):
            try:
                generation = self._generate(prompt)
                if self.cache is not None:
                    self.cache.put(prompt, generation)
                return generation, False, None
            except Exception as exc:  # noqa: BLE001 - archived, not silenced
                error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < self.params.llm_retries:
                    time.sleep(self.params.llm_retry_sleep_seconds)
        return "", False, error

    # -- per-method hooks ---------------------------------------------------
    def extract_rewrites(self, payload: dict) -> tuple[str, list[str]]:
        """Return (primary rewrite text, extra subquery channels)."""
        raise NotImplementedError

    def method_validate(self, payload: dict, query_text: str) -> list[str]:
        """Extra non-fatal flags for this method (override when needed)."""
        return []

    # -- validation ---------------------------------------------------------
    def _schema_check(self, payload: dict) -> str | None:
        for key in self.required_fields:
            if key not in payload:
                return f"missing_field:{key}"
        return None

    def _safety_flags(self, rewrite_texts: list[str],
                      query_text: str) -> list[str]:
        flags: list[str] = []
        joined = " ".join(rewrite_texts).lower()
        original = str(query_text).lower()
        for pattern in self.params.advice_patterns:
            if re.search(pattern, joined) and not re.search(pattern, original):
                flags.append(f"advice_language:{pattern}")
        for term in self.params.clinical_terms:
            if re.search(rf"\b{re.escape(term)}\b", joined) and not re.search(
                    rf"\b{re.escape(term)}\b", original):
                flags.append(f"new_clinical_term:{term}")
        rewrite_tokens = content_tokens(joined)
        if rewrite_tokens:
            novel = rewrite_tokens - content_tokens(original)
            fraction = len(novel) / len(rewrite_tokens)
            threshold = self.params.novel_token_thresholds.get(
                self.method_name, 1.0)
            if fraction > threshold:
                flags.append(
                    f"high_novel_token_fraction:{fraction:.2f}>{threshold}")
        return flags

    # -- main entry ----------------------------------------------------------
    def rewrite(self, query_id: str, query_text: str) -> RewriteResult:
        prompt = self.build_prompt(query_text)
        provenance = {
            "model_spec": self.params.llm_model,
            "prompt_name": self.prompt_name,
            "prompt_sha256": self._prompt_sha,
            "temperature": self.params.llm_temperature,
            "prompt_chars": len(prompt),
            # canonical llm_client returns text only; exact token usage is
            # not exposed, so archive a chars/4 estimate and say so.
            "prompt_tokens_estimated_chars_div4": len(prompt) // 4,
            "llm_calls": 0, "cache_hit": False, "json_repaired": False,
        }
        result = RewriteResult(
            query_id=str(query_id), original_query=str(query_text),
            rewrite_method=self.method_name, status="failed",
            provenance=provenance)

        generation, cache_hit, error = self._generate_cached(prompt)
        provenance["cache_hit"] = cache_hit
        provenance["llm_calls"] = 0 if cache_hit else 1
        if error is not None:
            result.failure_reason = f"llm_failure:{error}"
            return result

        payload = None
        try:
            payload = extract_json_object(generation)
        except ValueError:
            # One repair attempt with an explicit JSON-only reminder; the
            # repair prompt differs so the cache cannot replay the bad output.
            import configuration as _config
            repair_prompt = prompt + "\n\n" + _config.prompt("json_repair_suffix")
            generation2, cache_hit2, error2 = self._generate_cached(repair_prompt)
            provenance["llm_calls"] += 0 if cache_hit2 else 1
            provenance["json_repaired"] = True
            if error2 is not None:
                result.failure_reason = f"llm_failure:{error2}"
                return result
            try:
                payload = extract_json_object(generation2)
            except ValueError as exc:
                result.failure_reason = f"malformed_json:{exc}"
                return result
            generation = generation2
        provenance["generation_chars"] = len(generation)

        schema_error = self._schema_check(payload)
        if schema_error:
            result.payload = payload
            result.failure_reason = f"schema:{schema_error}"
            return result

        rewrite_text, subqueries = self.extract_rewrites(payload)
        rewrite_text = str(rewrite_text or "").strip()
        subqueries = [str(s).strip() for s in subqueries if str(s).strip()]
        subqueries = subqueries[:self.params.max_subqueries]
        if not rewrite_text and not subqueries:
            result.payload = payload
            result.failure_reason = "empty_rewrite"
            return result

        flags = self._safety_flags([rewrite_text, *subqueries], query_text)
        flags += self.method_validate(payload, query_text)
        result.payload = payload
        result.subqueries = subqueries
        result.validation_flags = flags
        fatal = [f for f in flags if f.startswith(("advice_language:",
                                                   "new_clinical_term:"))]
        if fatal:
            # docs_v2/48 §2.0: rewrites must not answer or add new
            # advice/diagnoses; hard-fail and keep the original channel.
            result.failure_reason = f"safety:{fatal[0]}"
            return result
        result.rewrite_text = rewrite_text
        result.status = "ok"
        return result

    def rewrite_batch(self, queries: list[dict]) -> list[RewriteResult]:
        return [self.rewrite(q["query_id"], q["query_text"]) for q in queries]


class FaithfulCompressionRewriter(BaseQueryRewriter):
    method_name = "compression"
    prompt_name = "query_rewrite_compression_v1"
    required_fields = ("compressed_query", "preserved_needs",
                       "preserved_constraints", "removed_details",
                       "possible_information_loss")

    def extract_rewrites(self, payload):
        return payload.get("compressed_query"), []

    def method_validate(self, payload, query_text):
        flags = []
        constraint_markers = re.search(
            r"\b(can't|cannot|can not|won't|without|not allowed|no longer|"
            r"already tried|tried every|doesn't work|didn't work)\b",
            str(query_text).lower())
        if constraint_markers and not payload.get("preserved_constraints"):
            flags.append("possible_constraint_loss")
        original_len = len(str(query_text))
        if original_len and len(str(payload.get("compressed_query") or "")) \
                > original_len:
            flags.append("not_actually_compressed")
        return flags


class NeedConstraintDecompositionRewriter(BaseQueryRewriter):
    method_name = "decomposition"
    prompt_name = "query_rewrite_decomposition_v1"
    required_fields = ("primary_need", "additional_needs", "constraints",
                       "failed_attempts", "desired_outcome",
                       "standalone_need_queries", "need_constraint_queries",
                       "ambiguities")

    def extract_rewrites(self, payload):
        standalone = payload.get("standalone_need_queries") or []
        queries = [item.get("query") for item in standalone
                   if isinstance(item, dict) and item.get("query")]
        primary = queries[0] if queries else ""
        extras = queries[1:] + [
            q for q in (payload.get("need_constraint_queries") or []) if q]
        return primary, extras

    def method_validate(self, payload, query_text):
        flags = []
        if not payload.get("primary_need"):
            flags.append("no_need_detected")
        standalone = payload.get("standalone_need_queries") or []
        ids = [item.get("need_id") for item in standalone
               if isinstance(item, dict)]
        if ids and ids[0] != "primary":
            flags.append("first_need_id_not_primary")
        if payload.get("constraints") and standalone and not any(
                item.get("relevant_constraints") for item in standalone
                if isinstance(item, dict)):
            flags.append("constraints_not_attached_to_any_need")
        return flags


class ConceptStepBackRewriter(BaseQueryRewriter):
    method_name = "concept_step_back"
    prompt_name = "query_rewrite_concept_stepback_v1"
    required_fields = ("concept_query", "concepts",
                       "unsafe_inferences_avoided")

    def extract_rewrites(self, payload):
        return payload.get("concept_query"), []

    def method_validate(self, payload, query_text):
        flags = []
        original = str(query_text).lower()
        concepts = payload.get("concepts") or []
        for item in concepts[:self.params.max_concepts]:
            if not isinstance(item, dict):
                continue
            span = str(item.get("source_span") or "")
            if span and span.lower() not in original:
                flags.append(
                    f"concept_span_not_in_original:{span[:40]}")
            ctype = item.get("type")
            if ctype not in {"problem", "constraint", "behaviour",
                             "environment", "outcome"}:
                flags.append(f"concept_type_invalid:{ctype}")
        if not concepts:
            flags.append("no_concepts")
        return flags


class ExpectedEvidenceRewriter(BaseQueryRewriter):
    method_name = "expected_evidence"
    prompt_name = "query_rewrite_expected_evidence_v1"
    required_fields = ("expected_evidence_query", "required_properties",
                       "required_constraints", "excluded_evidence_types")

    def extract_rewrites(self, payload):
        return payload.get("expected_evidence_query"), []


REWRITER_CLASSES = {
    cls.method_name: cls for cls in (
        FaithfulCompressionRewriter, NeedConstraintDecompositionRewriter,
        ConceptStepBackRewriter, ExpectedEvidenceRewriter)}


class QueryRewritePipeline:
    """Run selected rewriters over a query batch, emitting §2.0 channel rows.

    The ``original`` channel is always emitted first for every query, so a
    rewriter failure can never remove a query from downstream retrieval.
    """

    def __init__(self, params: QueryRewriteParameters, *,
                 methods: tuple[str, ...] = REWRITE_METHODS,
                 generate_fn=None, cache_dir: Path | None = None):
        self.params = params
        unknown = set(methods) - set(REWRITER_CLASSES)
        if unknown:
            raise ValueError(f"unknown rewrite methods: {sorted(unknown)}")
        self.rewriters = {}
        for method in methods:
            cache = None
            if cache_dir is not None:
                cache = GenerationCache(
                    Path(cache_dir) / f"{method}_generations.json",
                    params.llm_model)
            self.rewriters[method] = REWRITER_CLASSES[method](
                params, generate_fn=generate_fn, cache=cache)

    @staticmethod
    def _channel_row(query_id: str, original_query: str, method: str,
                     rewrite_text: str, metadata: dict,
                     diagnostics: dict) -> dict:
        row = {key: None for key in CHANNEL_ROW_FIELDS}
        row.update({
            "query_id": str(query_id), "original_query": str(original_query),
            "rewrite_method": method, "rewrite_text": rewrite_text,
            "rewrite_metadata": metadata,
            "graph_entry_status": "not_yet_run",
            "diagnostics": diagnostics,
        })
        return row

    def run(self, queries: list[dict]) -> tuple[list[dict], dict]:
        """Returns (channel rows, run stats)."""
        rows: list[dict] = []
        stats = {"queries": len(queries), "llm_calls": 0, "cache_hits": 0,
                 "failures": {}, "flags": {}, "methods": list(self.rewriters)}
        for query in queries:
            qid = str(query["query_id"])
            text = str(query["query_text"])
            rows.append(self._channel_row(
                qid, text, "original", text, {}, {"status": "ok"}))
            for method, rewriter in self.rewriters.items():
                result = rewriter.rewrite(qid, text)
                stats["llm_calls"] += result.provenance.get("llm_calls", 0)
                stats["cache_hits"] += int(
                    bool(result.provenance.get("cache_hit")))
                for flag in result.validation_flags:
                    key = flag.split(":", 1)[0]
                    stats["flags"][key] = stats["flags"].get(key, 0) + 1
                if result.status != "ok":
                    stats["failures"][method] = (
                        stats["failures"].get(method, 0) + 1)
                diagnostics = {
                    "status": result.status,
                    "failure_reason": result.failure_reason,
                    "validation_flags": result.validation_flags,
                    "provenance": result.provenance,
                }
                rows.append(self._channel_row(
                    qid, text, method,
                    result.rewrite_text if result.status == "ok" else "",
                    result.unified_metadata(), diagnostics))
                # Decomposition subqueries are their own channels (§2.0:
                # every rewrite channel embeds/retrieves independently).
                for index, subquery in enumerate(result.subqueries, 1):
                    if result.status != "ok":
                        break
                    rows.append(self._channel_row(
                        qid, text, f"{method}_sub{index}", subquery,
                        {"parent_method": method, "subquery_index": index},
                        {"status": "ok", "provenance": {
                            "derived_from": method}}))
        return rows, stats
