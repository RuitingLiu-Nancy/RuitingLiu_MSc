"""One-step IRCoT-style iterative retrieval over the Official HippoRAG2 runner.

Reproduction target: StonyBrookNLP/ircot (Trivedi et al., ACL 2023),
``commaqa/inference/ircot.py``.  The following pieces are direct ports of the
upstream code (same names, same logic):

- ``para_to_text``            <- upstream ``para_to_text``
- ``is_reasoning_sentence``   <- upstream ``is_reasoning_sentence``
- ``remove_reasoning_sentences`` <- upstream helper
- ``remove_wh_words``         <- upstream ``remove_wh_words``
- CoT prompt layout follows upstream ``StepByStepCOTGenParticipant.query``; the provider-visible wrapper is supplied outside this release.
- next-query rule             <- upstream ``RetrieveAndResetParagraphsParticipant``
  with ``query_source="question_or_last_generated_sentence"``: the next query is
  the last generated non-reasoning CoT sentence; if none survives, the original
  question (which makes round two a no-op by design and is archived as such).
- termination rule            <- upstream ``answer_extractor_regex``
  ``".* answer is (.*)"`` on the newly generated sentence.
- merge rule (``cumulate``)   <- upstream cumulative paragraph collection:
  round-one results first, then unseen round-two results appended, capped.

Declared deviations from upstream (recorded in docs_v2/11 §F):

1. Retriever: upstream steps retrieve with BM25/Elasticsearch; here BOTH rounds
   use the same frozen Official HippoRAG2 retriever (same graph.pickle, corpus,
   recognition, seeds), which is this project's fixed-graph requirement.
2. Sentence splitting: upstream uses spaCy ``en_core_web_sm``; this module uses
   a deterministic regex splitter to avoid a new model dependency.
3. Few-shot demonstrations: upstream reads dataset-specific annotated CoT
   demos; this domain has none, so the default prompt header is a fixed
   zero-shot instruction (loaded from the external prompt store). A demos file can be supplied via
   ``prompt_header_path`` without code changes.
4. ``interleave`` merge option: upstream's readers consume the collected
   paragraph SET, so its append order carries no ranking semantics; our
   downstream candidate gates read ranked prefixes (top-10/20), so the default
   merge alternates round-one/round-two ranks.  ``merge_strategy="cumulate"``
   restores the exact upstream behaviour.

No graph, corpus, seed, restart-vector or recognition component is touched:
this stage only issues one extra LLM call per query and one extra retrieval
pass through the unchanged official pipeline.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


# Upstream: prompts are joined to the test example with a triple newline.
PROMPT_JOINER = "\n\n\n"


def para_to_text(title: str, para: str, max_num_words: int) -> str:
    """Direct port of upstream ``para_to_text`` (word cap, then title header)."""
    para = " ".join(para.split(" ")[:max_num_words])
    para = (
        para.strip()
        if para.strip().startswith("Wikipedia Title: ")
        else "Wikipedia Title: " + title + "\n" + para.strip()
    )
    return para


def is_reasoning_sentence(sentence: str) -> bool:
    """Direct port of upstream ``is_reasoning_sentence``."""
    starters = ["thus ", "thus,", "so ", "so,", "that is,", "therefore", "hence"]
    for starter in starters:
        if sentence.lower().startswith(starter):
            return True
    regex = re.compile(r"(.*)(\d[\d,]*\.?\d+|\d+) ([+-]) (\d[\d,]*\.?\d+|\d+) = (\d[\d,]*\.?\d+|\d+)(.*)")
    return bool(re.match(regex, sentence))


def remove_reasoning_sentences(sentences: list[str]) -> list[str]:
    """Direct port of upstream ``remove_reasoning_sentences``."""
    return [sentence for sentence in sentences if not is_reasoning_sentence(sentence)]


def remove_wh_words(text: str) -> str:
    """Direct port of upstream ``remove_wh_words`` (upstream applies it to BM25
    queries only; kept optional here because the official retriever is not BM25)."""
    wh_words = {"who", "what", "when", "where", "why", "which", "how", "does", "is"}
    words = [word for word in text.split(" ") if word.strip().lower() not in wh_words]
    return " ".join(words)


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def first_sentence(text: str) -> str:
    """First sentence of a generation (deviation #2: regex, not spaCy).

    Upstream keeps only ``spacy(new_generation).sents[0]``; this mirrors that
    contract: leading whitespace stripped, split at the first terminal
    punctuation followed by a sentence-initial character.
    """
    text = text.strip()
    if not text:
        return ""
    # Generation may start with a spurious "A:" continuation marker.
    if text.startswith("A:"):
        text = text[2:].strip()
    parts = _SENTENCE_BOUNDARY.split(text, maxsplit=1)
    return parts[0].strip()


@dataclass(frozen=True)
class IRCoTParameters:
    """All values are read from configuration/params.yaml::official_ircot."""
    prompt_context_count: int = 8          # paragraphs shown to the CoT LLM
    max_para_num_words: int = 350          # upstream default
    answer_extractor_regex: str = ".* answer is (.*)"  # upstream default
    merge_strategy: str = "interleave"     # "interleave" | "cumulate" (upstream)
    query_transform: str = "none"          # "none" | "remove_wh_words" (upstream/BM25)
    instruction: str | None = None
    prompt_header_path: str | None = None  # optional few-shot demos file
    llm_max_tokens: int = 200
    llm_temperature: float = 0.0
    llm_retries: int = 3
    llm_retry_sleep_seconds: float = 5.0


def build_cot_prompt(question: str, titles: list[str], paras: list[str],
                     params: IRCoTParameters, generation_so_far: str = "") -> str:
    """Upstream ``StepByStepCOTGenParticipant`` prompt layout, one-step case."""
    shown = list(zip(titles, paras))[:params.prompt_context_count]
    context = "\n\n".join(
        para_to_text(title, para, params.max_para_num_words) for title, para in shown)
    import configuration as config
    if params.prompt_header_path:
        header = Path(params.prompt_header_path).read_text(encoding="utf-8").strip()
    elif params.instruction:
        header = params.instruction
    else:
        header = config.prompt("ircot_zero_shot_instruction")
    template = config.prompt("ircot_request_wrapper")
    return template.format(
        instruction=header,
        context=context,
        question=question,
        generation_so_far=generation_so_far,
    ).strip()


def plan_second_query(generation: str, question: str,
                      params: IRCoTParameters) -> dict:
    """Upstream next-step logic: sentence -> terminate / reasoning-skip / query."""
    sentence = first_sentence(generation)
    answer_regex = re.compile(params.answer_extractor_regex)
    if not sentence:
        return {"cot_sentence": "", "second_query": None,
                "status": "empty_generation"}
    match = answer_regex.match(sentence)
    if match:
        return {"cot_sentence": sentence, "second_query": None,
                "status": "terminated_answer", "extracted_answer": match.group(1)}
    surviving = remove_reasoning_sentences([sentence])
    if not surviving:
        # Upstream falls back to the original question, i.e. round two would
        # re-run round one.  Archive it as a skip instead of paying for a
        # provably identical retrieval.
        return {"cot_sentence": sentence, "second_query": question,
                "status": "reasoning_sentence_fallback_to_question"}
    query = surviving[-1]
    if params.query_transform == "remove_wh_words":
        query = remove_wh_words(query)
    elif params.query_transform != "none":
        raise ValueError(f"unknown query_transform: {params.query_transform}")
    return {"cot_sentence": sentence, "second_query": query, "status": "second_query"}


def merge_rankings(first_ids: list[str], second_ids: list[str], top_k: int,
                   strategy: str) -> tuple[list[str], dict]:
    """Merge two ranked comment-id lists without scores (rank-only contract).

    ``cumulate``  : upstream-faithful append of unseen round-two ids.
    ``interleave``: alternate ranks r1[0], r2[0], r1[1], r2[2], ... (deviation #4).
    """
    if strategy not in {"cumulate", "interleave"}:
        raise ValueError(f"unknown merge strategy: {strategy}")
    merged: list[str] = []
    seen: set[str] = set()

    def add(cid: str) -> None:
        if cid not in seen:
            seen.add(cid)
            merged.append(cid)

    if strategy == "cumulate":
        for cid in first_ids:
            add(cid)
        for cid in second_ids:
            add(cid)
    else:
        i = j = 0
        while (i < len(first_ids) or j < len(second_ids)) and len(merged) < top_k:
            if i < len(first_ids):
                add(first_ids[i]); i += 1
            if j < len(second_ids):
                add(second_ids[j]); j += 1
    merged = merged[:top_k]
    second_exclusive = [cid for cid in merged if cid not in set(first_ids)]
    return merged, {
        "merge_strategy": strategy,
        "merged_count": len(merged),
        "second_round_new_in_merged": len(second_exclusive),
        "second_round_new_ids_top20": [
            cid for cid in merged[:20] if cid not in set(first_ids)],
    }


class GenerationCache:
    """Content-addressed cache for CoT generations (prompt+model keyed).

    Stored as one JSON file per run cache dir; re-runs with identical prompts
    and model make zero LLM calls.  Never caches failures.
    """

    def __init__(self, path: Path, model_spec: str):
        self.path = Path(path)
        self.model_spec = model_spec
        self._data: dict[str, str] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        self.hits = 0
        self.misses = 0

    def key(self, prompt: str) -> str:
        digest = hashlib.sha256()
        digest.update(self.model_spec.encode()); digest.update(b"\0")
        digest.update(prompt.encode())
        return digest.hexdigest()

    def get(self, prompt: str) -> str | None:
        value = self._data.get(self.key(prompt))
        if value is not None:
            self.hits += 1
        return value

    def put(self, prompt: str, generation: str) -> None:
        self.misses += 1
        self._data[self.key(prompt)] = generation
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=0), encoding="utf-8")


def run_one_step_ircot(
    queries: list[dict],
    first_rankings: dict[str, list[str]],
    first_titles_texts: dict[str, tuple[list[str], list[str]]],
    *,
    params: IRCoTParameters,
    top_k: int,
    generate_fn,
    retrieve_second_fn,
    cache: GenerationCache | None = None,
) -> tuple[dict[str, list[str]], list[dict], dict]:
    """Execute the full one-step loop for every query.

    ``generate_fn(prompt) -> str`` performs one LLM call (injected so tests run
    without any API).  ``retrieve_second_fn(second_queries: list[str]) ->
    dict[str, list[str]]`` runs the SAME official retriever on the archived
    second queries and returns ranked comment ids per second-query string.

    Returns (merged rankings by query_id, per-query trace rows, runtime stats).
    """
    started = time.perf_counter()
    plans: dict[str, dict] = {}
    llm_failures = 0
    for query in queries:
        qid, question = str(query["query_id"]), str(query["question"])
        titles, paras = first_titles_texts[qid]
        prompt = build_cot_prompt(question, titles, paras, params)
        generation, cache_hit, error = None, False, None
        if cache is not None:
            cached = cache.get(prompt)
            if cached is not None:
                generation, cache_hit = cached, True
        if generation is None:
            for attempt in range(params.llm_retries):
                try:
                    generation = generate_fn(prompt)
                    break
                except Exception as exc:  # noqa: BLE001 - archived, not silenced
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt + 1 < params.llm_retries:
                        time.sleep(params.llm_retry_sleep_seconds)
            if generation is not None and cache is not None:
                cache.put(prompt, generation)
        if generation is None:
            llm_failures += 1
            plans[qid] = {"cot_sentence": "", "second_query": None,
                          "status": "llm_failure", "error": error,
                          "prompt_chars": len(prompt), "llm_cache_hit": False}
            continue
        plan = plan_second_query(generation, question, params)
        plan["prompt_chars"] = len(prompt)
        plan["llm_cache_hit"] = cache_hit
        plan["generation"] = generation[:600]
        plans[qid] = plan

    # One batched second retrieval for all queries that produced a genuinely
    # new second query (skips: terminated / failure / question fallback).
    second_queries: list[str] = []
    for qid, plan in plans.items():
        if plan["status"] == "second_query":
            second_queries.append(plan["second_query"])
    second_queries = list(dict.fromkeys(second_queries))
    second_results = retrieve_second_fn(second_queries) if second_queries else {}

    merged_rankings: dict[str, list[str]] = {}
    trace_rows: list[dict] = []
    for query in queries:
        qid, question = str(query["query_id"]), str(query["question"])
        plan = plans[qid]
        first_ids = list(first_rankings[qid])
        second_ids: list[str] = []
        if plan["status"] == "second_query":
            second_ids = list(second_results.get(plan["second_query"], []))
        merged, merge_diag = merge_rankings(
            first_ids, second_ids, top_k, params.merge_strategy)
        if not second_ids:
            merged = first_ids[:top_k]
            merge_diag = {"merge_strategy": params.merge_strategy,
                          "merged_count": len(merged),
                          "second_round_new_in_merged": 0,
                          "second_round_new_ids_top20": [],
                          "fallback_to_first_round": True}
        merged_rankings[qid] = merged
        trace_rows.append({
            "query_id": qid, "query_text": question,
            **{k: plan.get(k) for k in (
                "status", "cot_sentence", "second_query", "extracted_answer",
                "error", "prompt_chars", "llm_cache_hit", "generation")},
            "first_round_count": len(first_ids),
            "second_round_count": len(second_ids),
            "second_query_identical_to_original": (
                plan.get("second_query") == question),
            **merge_diag,
        })
    stats = {
        "queries": len(queries),
        "llm_calls": (cache.misses if cache is not None else
                      sum(1 for p in plans.values()
                          if not p.get("llm_cache_hit") and p["status"] != "llm_failure")),
        "llm_cache_hits": cache.hits if cache is not None else 0,
        "llm_failures": llm_failures,
        "second_retrieval_queries": len(second_queries),
        "status_counts": {
            status: sum(1 for p in plans.values() if p["status"] == status)
            for status in sorted({p["status"] for p in plans.values()})},
        "seconds": time.perf_counter() - started,
    }
    return merged_rankings, trace_rows, stats
