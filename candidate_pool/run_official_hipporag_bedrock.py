#!/usr/bin/env python3
# 【文件 045】包装固定版本官方 HippoRAG2，配置图入口、识别、重启和输出
# 【流程位置】候选访问与图检索；所属包：candidate_pool
# 【主要函数】run, main
# 【输入接口】path, query, scores 等函数参数；返回值及写出操作见对应函数
# 【依赖文件】configuration/__init__.py

"""Build/reuse the pinned upstream HippoRAG2 index and run the final graph routes.

The two routes share OpenIE extraction, linking, graph construction and static
PPR. They differ only in direct passage restart. Historical ontology and
query-time experiments are preserved on the development archive branch.
The filename is retained for compatibility; runtime model/provider identities
are explicit arguments and recorded in the output manifest.
"""
from __future__ import annotations

import argparse
import ast
import functools
import json
import os
import platform
import resource
import time
from datetime import datetime, timezone
from hashlib import md5, sha256
from pathlib import Path
from types import MethodType


RETRIEVAL_PROFILES = ("no_recognition", "fact_only_no_recognition")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _l2_normalize_rows(values):
    """Return float32 row-normalized vectors and the zero-row count.

    HippoRAG's local Transformers adapter returns raw SentenceTransformer
    vectors, while both the frozen SBERT baseline and sentence-transformers'
    cosine helper normalize rows.  Keeping this conversion pure makes the
    backend contract directly testable.
    """
    import numpy as np

    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError(f"expected a 2-D embedding matrix, got shape={rows.shape}")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    zero_count = int(np.sum(norms[:, 0] == 0))
    return rows / np.maximum(norms, 1e-12), zero_count


def _e5_role_prefix_texts(texts, *, instruction, query_instructions):
    """Apply the official E5 prefix for the HippoRAG retrieval role.

    HippoRAG marks query-to-fact and query-to-passage calls with an
    ``instruction`` value, while its indexed passage/entity/fact stores omit
    that value.  This follows the same role boundary as the upstream Cohere
    adapter without changing stored text or document identities.
    """
    if isinstance(texts, str):
        texts = [texts]
    role = "query" if instruction in query_instructions else "passage"
    prefix = "query: " if role == "query" else "passage: "
    prefixed = []
    already_prefixed = 0
    for text in texts:
        value = str(text)
        if value.startswith(prefix):
            already_prefixed += 1
            prefixed.append(value)
        else:
            prefixed.append(prefix + value)
    return prefixed, role, already_prefixed


def _install_llm_inference_blocker(hipporag) -> dict:
    """Make a retrieval run fail before any LLM inference can leave the host.

    Constructing HippoRAG still creates its configured client, but every
    callable inference entry point on the shared LLM/OpenIE object is replaced
    before ``index`` or ``retrieve`` runs.  This is intended for frozen-cache,
    recognition-free evaluations where a cache miss must be an error rather
    than an implicit provider call.
    """
    audit = {"enabled": True, "blocked_methods": [], "attempted_calls": 0}

    def blocked(*_args, **_kwargs):
        audit["attempted_calls"] += 1
        raise RuntimeError(
            "LLM inference is forbidden for this frozen-cache retrieval run"
        )

    targets = []
    for candidate in (
        getattr(hipporag, "llm_model", None),
        getattr(getattr(hipporag, "openie", None), "llm_model", None),
    ):
        if candidate is not None and all(candidate is not item for item in targets):
            targets.append(candidate)
    for target in targets:
        for method_name in ("infer", "batch_infer"):
            if callable(getattr(target, method_name, None)):
                setattr(target, method_name, blocked)
                audit["blocked_methods"].append(
                    f"{type(target).__name__}.{method_name}"
                )
    if not any(name.endswith(".infer") for name in audit["blocked_methods"]):
        raise RuntimeError("could not install the required LLM inference blocker")
    return audit


def _git_provenance() -> dict:
    """Best-effort code-version stamp for run manifests (audit blocker D-1).

    Never raises: environments without git (or with a locked .git) still get a
    manifest, but the missing stamp is recorded explicitly instead of silently.
    """
    import subprocess
    repo_root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
            text=True, timeout=10, check=True).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True,
            text=True, timeout=30, check=True).stdout
        return {"git_commit": commit,
                "git_dirty": bool(status.strip()),
                "git_dirty_paths": len(status.strip().splitlines())}
    except Exception as exc:  # noqa: BLE001 - archived, not silenced
        return {"git_commit": None, "git_dirty": None,
                "git_error": f"{type(exc).__name__}: {exc}"}


def _validate_query_contract(
    path: Path,
    *,
    test_split_used: bool,
    expected_query_sha256: str | None,
    expected_query_count: int | None,
) -> dict:
    """Fail closed before a frozen test adapter can enter retrieval.

    Validation callers retain the historical permissive contract.  A caller
    that explicitly declares a frozen test split must also bind the exact file
    bytes and row count; neither filename heuristics nor a manifest written
    after retrieval are sufficient provenance controls.
    """
    if test_split_used and (
        not expected_query_sha256 or expected_query_count is None
    ):
        raise ValueError(
            "test_split_used requires expected_query_sha256 and "
            "expected_query_count"
        )
    actual_sha256 = _sha256_file(path)
    if expected_query_sha256 and actual_sha256 != expected_query_sha256:
        raise ValueError(
            f"query adapter sha256 mismatch: {actual_sha256} != "
            f"{expected_query_sha256}"
        )
    return {
        "expected_query_sha256": expected_query_sha256,
        "actual_query_sha256": actual_sha256,
        "expected_query_count": expected_query_count,
    }


def _read_validation_adapter(
    path: Path, *, allow_frozen_test: bool = False,
) -> list[dict]:
    if "test" in path.name.lower() and not allow_frozen_test:
        raise ValueError("test split is frozen; official HippoRAG runner accepts validation only")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected a JSON list: {path}")
    return rows


def _supporting_paragraphs(query: dict) -> list[dict]:
    """Normalize supporting passages across project and public benchmarks."""
    paragraphs = query.get("paragraphs") or []
    if paragraphs:
        # Project adapters omit the flag because every listed paragraph is
        # gold; MuSiQue explicitly marks distractors as false.
        return [p for p in paragraphs if p.get("is_supporting") is not False]
    if query.get("contexts"):
        return [p for p in query["contexts"] if p.get("is_supporting")]
    return []


def _optional_gold_texts(query: dict) -> list[str]:
    """Return supporting paragraph text when present, otherwise empty slots."""
    return [
        str(p.get("text") or p.get("paragraph_text") or "")
        for p in _supporting_paragraphs(query)
    ]


def _gold_titles(query: dict) -> list[str]:
    """Use the same supporting-title convention as upstream ``main.py``."""
    if query.get("supporting_facts") and query.get("context"):
        supporting = {str(item[0]) for item in query["supporting_facts"]}
        return [str(item[0]) for item in query["context"] if str(item[0]) in supporting]
    return [str(p.get("title") or "") for p in _supporting_paragraphs(query)]


def _adapter_query_id(row: dict) -> str:
    """Normalize ``dataset/query.json`` to the project query id."""
    return Path(str(row.get("id") or row.get("_id") or "")).stem


def _top_embedding_facts(self, query_fact_scores):
    """Return upstream top-k embedding facts without applying recognition."""
    if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
        return [], []
    top_k = int(self.global_config.linking_top_k)
    indices = sorted(
        range(len(query_fact_scores)),
        key=lambda idx: float(query_fact_scores[idx]),
        reverse=True,
    )[:top_k]
    ids = [self.fact_node_keys[idx] for idx in indices]
    rows = self.fact_embedding_store.get_rows(ids)
    facts = [ast.literal_eval(rows[fact_id]["content"]) for fact_id in ids]
    return indices, facts


def _fact_endpoint_trace(hipporag, fact_indices, facts, query_fact_scores) -> list[dict]:
    """Expose the exact fact endpoints that seed upstream graph search."""
    rows = []
    for rank, (idx, fact) in enumerate(zip(fact_indices, facts, strict=True), 1):
        score = float(query_fact_scores[idx])
        endpoints = []
        for phrase in (str(fact[0]).lower(), str(fact[2]).lower()):
            # Exact upstream ``compute_mdhash_id`` contract, kept local so the
            # read-only trace core remains unit-testable without HippoRAG.
            entity_id = "entity-" + md5(phrase.encode()).hexdigest()
            document_frequency = len(hipporag.ent_node_to_chunk_ids.get(entity_id, set()))
            endpoints.append({
                "phrase": phrase,
                "entity_id": entity_id,
                "document_frequency": int(document_frequency),
                "specificity_adjusted_input": (
                    score / document_frequency if document_frequency else None
                ),
            })
        rows.append({
            "selected_rank": rank,
            "fact_index": int(idx),
            "fact": list(fact),
            "fact_score_entering_graph": score,
            "endpoints": endpoints,
        })
    return rows


def _install_entry_trace(hipporag, sink: dict[str, dict], *, top_k: int = 20) -> None:
    """Attach read-only query→fact/passage entry tracing to upstream methods.

    The wrappers call the already-installed retrieval profile and never alter
    returned values.  They reveal observed retrieval provenance, not a hidden
    reasoning chain.
    """
    if top_k < 1:
        raise ValueError("entry trace top_k must be positive")
    original_get_fact_scores = hipporag.get_fact_scores
    original_rerank_facts = hipporag.rerank_facts
    original_dense = hipporag.dense_passage_retrieval

    def _get_fact_scores(self, query):
        scores = original_get_fact_scores(query)
        record = sink.setdefault(query, {"query_text": query})
        if len(scores) and len(self.fact_node_keys):
            indices = sorted(
                range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True
            )[:top_k]
            ids = [self.fact_node_keys[idx] for idx in indices]
            facts_by_id = self.fact_embedding_store.get_rows(ids)
            record["candidate_facts"] = [{
                "embedding_rank": rank,
                "fact_index": int(idx),
                "fact_id": fact_id,
                "fact": list(ast.literal_eval(facts_by_id[fact_id]["content"])),
                "embedding_score": float(scores[idx]),
            } for rank, (idx, fact_id) in enumerate(zip(indices, ids, strict=True), 1)]
        else:
            record["candidate_facts"] = []
        return scores

    def _rerank_facts(self, query, query_fact_scores):
        indices, facts, log = original_rerank_facts(query, query_fact_scores)
        record = sink.setdefault(query, {"query_text": query})
        record["selected_fact_count"] = len(indices)
        record["selected_fact_seeds"] = _fact_endpoint_trace(
            self, indices, facts, query_fact_scores)
        record["selection_log"] = {
            key: value for key, value in log.items()
            if key not in {"facts_before_rerank", "facts_after_rerank"}
        }
        return indices, facts, log

    def _dense_passage_retrieval(self, query):
        sorted_ids, sorted_scores = original_dense(query)
        record = sink.setdefault(query, {"query_text": query})
        record["dense_passage_entry"] = [{
            "dense_rank": rank,
            "passage_index": int(idx),
            "passage_id": self.passage_node_keys[int(idx)],
            "dense_score": float(score),
        } for rank, (idx, score) in enumerate(
            zip(sorted_ids[:top_k], sorted_scores[:top_k], strict=True), 1)]
        return sorted_ids, sorted_scores

    hipporag.get_fact_scores = MethodType(_get_fact_scores, hipporag)
    hipporag.rerank_facts = MethodType(_rerank_facts, hipporag)
    hipporag.dense_passage_retrieval = MethodType(_dense_passage_retrieval, hipporag)


def _install_retrieval_profile(hipporag, profile: str) -> dict:
    """Preserve upstream fact similarity weights and bypass recognition.

    Passage-restart Graph keeps the upstream direct passage restart weight.
    Fact-only Graph sets that weight to zero; both use the same indexed graph.
    """
    if profile not in RETRIEVAL_PROFILES:
        raise ValueError(f"unknown retrieval profile: {profile}")
    if profile == "fact_only_no_recognition":
        hipporag.global_config.passage_node_weight = 0.0

    def _bypass_recognition(self, query, query_fact_scores):
        del query
        indices, facts = _top_embedding_facts(self, query_fact_scores)
        return indices, facts, {
            "facts_before_rerank": facts,
            "facts_after_rerank": facts,
            "ablation": "recognition_filter_bypassed",
            "fact_seed_weighting": "similarity",
            "selected_fact_weights": [float(query_fact_scores[i]) for i in indices],
        }

    hipporag.rerank_facts = MethodType(_bypass_recognition, hipporag)
    return {
        "profile": profile,
        "graph_construction_changed": False,
        "recognition_filter_enabled": False,
        "recognition_filter_mode": "bypassed",
        "dense_passage_teleport_enabled": profile != "fact_only_no_recognition",
        "ppr_enabled": True,
        "fact_seed_weighting": "similarity",
    }


def run(
    corpus_path: Path,
    queries_path: Path,
    save_dir: Path,
    out_path: Path,
    max_docs: int,
    max_queries: int,
    llm_model: str,
    embedding_model: str,
    top_k: int,
    openie_workers: int,
    normalize_transformer_embeddings: bool = False,
    e5_role_prefixes: bool = False,
    retrieval_profile: str = "no_recognition",
    entry_trace: bool = False,
    entry_trace_top_k: int = 20,
    prepend_title: bool = False,
    openie_cache_provenance: str | None = None,
    test_split_used: bool = False,
    expected_query_sha256: str | None = None,
    expected_query_count: int | None = None,
    forbid_llm_inference: bool = False,
) -> dict:
    if retrieval_profile not in RETRIEVAL_PROFILES:
        raise ValueError(f"unknown retrieval profile: {retrieval_profile}")
    run_started = time.perf_counter()
    rss_started = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    corpus = _read_validation_adapter(corpus_path)
    query_contract = _validate_query_contract(
        queries_path,
        test_split_used=test_split_used,
        expected_query_sha256=expected_query_sha256,
        expected_query_count=expected_query_count,
    )
    queries = _read_validation_adapter(
        queries_path, allow_frozen_test=test_split_used)
    if expected_query_count is not None and len(queries) != expected_query_count:
        raise ValueError(
            f"query adapter count mismatch: {len(queries)} != "
            f"{expected_query_count}"
        )
    if test_split_used and max_queries != expected_query_count:
        raise ValueError(
            "frozen test retrieval must request the complete expected query count"
        )
    if max_docs < 1 or max_queries < 1:
        raise ValueError("--max-docs and --max-queries must both be positive")
    corpus = corpus[:max_docs]
    queries = queries[:max_queries]
    texts = []
    for row in corpus:
        body = str(row.get("text") or "").strip()
        title = str(row.get("title") or "").strip()
        texts.append(f"{title}\n{body}" if prepend_title and title else body)
    if not all(texts):
        raise ValueError("corpus contains an empty text field")

    # Upstream main currently exposes a module for ``from hipporag import
    # HippoRAG``; importing the class directly is the smallest environment
    # compatibility shim and does not alter the retrieval algorithm.
    from hipporag.HippoRAG import HippoRAG

    if openie_workers < 1:
        raise ValueError("--openie-workers must be positive")
    if e5_role_prefixes:
        if embedding_model != "Transformers/intfloat/e5-base-v2":
            raise ValueError(
                "--e5-role-prefixes requires "
                "--embedding-model Transformers/intfloat/e5-base-v2"
            )
        if not normalize_transformer_embeddings:
            raise ValueError(
                "--e5-role-prefixes requires --normalize-transformer-embeddings"
            )
    # Upstream online OpenIE uses ThreadPoolExecutor() without an explicit
    # bound (32 workers on this host), which exceeds the account's Bedrock
    # on-demand request quota.  Limit only transport concurrency; prompts,
    # extraction, linking, graph construction and PPR remain upstream code.
    import hipporag.information_extraction.openie_openai as openie_module
    openie_module.ThreadPoolExecutor = functools.partial(
        openie_module.ThreadPoolExecutor, max_workers=openie_workers)

    save_dir.mkdir(parents=True, exist_ok=True)
    # Cache-only retrieval still has to construct HippoRAG's OpenAI-compatible
    # client before the inference blocker below can be installed.  Recent
    # OpenAI SDK versions reject that constructor when no credential is
    # present, even though constructing the client makes no request.  Supply a
    # deliberately invalid, process-local placeholder only for the explicit
    # fail-closed mode, overriding any inherited credential.  Every callable
    # inference entry point is replaced immediately afterwards, so a cache
    # miss remains a hard local error.
    if forbid_llm_inference:
        os.environ["OPENAI_API_KEY"] = "cache-only-inference-is-forbidden"
    hipporag = HippoRAG(
        save_dir=str(save_dir),
        llm_model_name=llm_model,
        embedding_model_name=embedding_model,
    )
    llm_inference_audit = (
        _install_llm_inference_blocker(hipporag)
        if forbid_llm_inference
        else {"enabled": False, "blocked_methods": [], "attempted_calls": 0}
    )
    if forbid_llm_inference:
        llm_inference_audit["credential_mode"] = "invalid_local_placeholder"

    # Declared shims at the embedding boundary. Cohere has a provider-specific
    # character cap. The local Transformers backend may optionally be L2
    # normalized so its dot product is exactly the cosine used by the frozen
    # Sentence-BERT baseline. Neither shim changes OpenIE, graph propagation,
    # document identities or returned text.
    _COHERE_CHAR_CAP = 2048
    _trunc_stats = {"n_truncated": 0, "n_encoded": 0}
    _normalization_stats = {"n_normalized": 0, "zero_norm": 0}
    _prefix_stats = {
        "enabled": bool(e5_role_prefixes),
        "query_prefix": "query: " if e5_role_prefixes else None,
        "passage_prefix": "passage: " if e5_role_prefixes else None,
        "query_inputs": 0,
        "passage_inputs": 0,
        "already_prefixed_inputs": 0,
    }
    _orig_encode = hipporag.embedding_model.encode
    _orig_batch_encode = hipporag.embedding_model.batch_encode

    def _capped_encode(texts_in, *a, **k):
        if isinstance(texts_in, str):  # defensive: never iterate a str by char
            texts_in = [texts_in]
        capped = []
        for t in texts_in:
            s = str(t)
            _trunc_stats["n_encoded"] += 1
            if len(s) > _COHERE_CHAR_CAP:
                _trunc_stats["n_truncated"] += 1
                s = s[:_COHERE_CHAR_CAP]
            capped.append(s)
        return _orig_encode(capped, *a, **k)

    def _normalized_encode(texts_in, *a, **k):
        if isinstance(texts_in, str):
            texts_in = [texts_in]
        values, zero_count = _l2_normalize_rows(
            _orig_encode(texts_in, *a, **k))
        _normalization_stats["n_normalized"] += int(values.shape[0])
        _normalization_stats["zero_norm"] += zero_count
        return values

    if "cohere" in embedding_model.lower():
        hipporag.embedding_model.encode = _capped_encode
    elif normalize_transformer_embeddings:
        if not embedding_model.startswith("Transformers/"):
            raise ValueError(
                "--normalize-transformer-embeddings requires Transformers/ backend"
            )
        hipporag.embedding_model.encode = _normalized_encode

    if e5_role_prefixes:
        query_instructions = set(hipporag.embedding_model.search_query_instr)

        def _role_prefixed_batch_encode(texts_in, *a, **k):
            prefixed, role, already_prefixed = _e5_role_prefix_texts(
                texts_in,
                instruction=k.get("instruction"),
                query_instructions=query_instructions,
            )
            _prefix_stats[f"{role}_inputs"] += len(prefixed)
            _prefix_stats["already_prefixed_inputs"] += already_prefixed
            return _orig_batch_encode(prefixed, *a, **k)

        hipporag.embedding_model.batch_encode = _role_prefixed_batch_encode

    hipporag.index(docs=texts)
    profile_manifest = _install_retrieval_profile(hipporag, retrieval_profile)
    entry_trace_by_query: dict[str, dict] = {}
    if entry_trace:
        _install_entry_trace(
            hipporag, entry_trace_by_query, top_k=entry_trace_top_k)
    query_texts = [str(row["question"]) for row in queries]
    solutions = hipporag.retrieve(query_texts, num_to_retrieve=top_k)
    # Map retrieved texts back to comment ids so downstream scoring can use
    # ir_metrics against gold_comment_ids (texts alone are not identifiers).
    text_to_title: dict[str, str] = {}
    duplicate_texts = 0
    for row, indexed_text in zip(corpus, texts, strict=True):
        t = indexed_text
        if t in text_to_title:
            duplicate_texts += 1
        else:
            text_to_title[t] = str(row.get("title") or "")

    rows = []
    for query, solution in zip(queries, solutions, strict=True):
        docs_list = list(solution.docs)
        rows.append({
            "query_id": _adapter_query_id(query),
            "query_text": str(query["question"]),
            "retrieved_titles": [text_to_title.get(str(t), "") for t in docs_list],
            "retrieved_texts": docs_list,
            "retrieved_scores": [float(x) for x in solution.doc_scores],
            "gold_titles": _gold_titles(query),
            # External adapters such as MuSiQue may retain only supporting
            # titles because retrieval scoring is title-based.  Text is useful
            # metadata when present, but must not be a required output field.
            "gold_texts": _optional_gold_texts(query),
        })
        if entry_trace:
            trace = entry_trace_by_query.setdefault(
                str(query["question"]), {"query_text": str(query["question"])})
            trace["query_id"] = _adapter_query_id(query)
            trace["retrieved_titles"] = rows[-1]["retrieved_titles"][:entry_trace_top_k]
            trace["retrieved_scores"] = rows[-1]["retrieved_scores"][:entry_trace_top_k]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    entry_trace_path = out_path.with_suffix(".entry_trace.jsonl")
    if entry_trace:
        with entry_trace_path.open("w", encoding="utf-8") as fh:
            for query in queries:
                trace = entry_trace_by_query.get(str(query["question"]), {})
                fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
    manifest = {
        "protocol": "final HippoRAG2 fixed-graph retrieval",
        "method_version": "official-fixed-graph-query-time-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **_git_provenance(),
        "random_seed": None,
        "test_split_used": bool(test_split_used),
        "upstream": "OSU-NLP-Group/HippoRAG@ad30fc3e2062202d9e975e32cd28212424a56ccb",
        "corpus_path": str(corpus_path),
        "queries_path": str(queries_path),
        "corpus_sha256": _sha256_file(corpus_path),
        "query_split_sha256": _sha256_file(queries_path),
        "query_contract": query_contract,
        "indexed_documents": len(texts),
        "retrieved_queries": len(rows),
        "top_k": top_k,
        "llm_model": llm_model,
        "llm_inference_audit": llm_inference_audit,
        "openie_cache_provenance": openie_cache_provenance or llm_model,
        "embedding_model": embedding_model,
        "transformer_embeddings_l2_normalized":
            normalize_transformer_embeddings,
        "embedding_vectors_normalized":
            _normalization_stats["n_normalized"],
        "embedding_zero_norm_vectors":
            _normalization_stats["zero_norm"],
        "e5_role_prefixes": _prefix_stats,
        "openie_workers": openie_workers,
        "retrieval_ablation": profile_manifest,
        "entry_trace": str(entry_trace_path) if entry_trace else None,
        "entry_trace_queries": len(entry_trace_by_query) if entry_trace else 0,
        "entry_trace_top_k": entry_trace_top_k if entry_trace else None,
        "duplicate_corpus_texts": duplicate_texts,
        "document_construction": "title\\ntext" if prepend_title else "text",
        "embedding_char_cap":
            _COHERE_CHAR_CAP if "cohere" in embedding_model.lower() else None,
        "embedding_inputs_truncated": _trunc_stats["n_truncated"],
        "embedding_inputs_total": _trunc_stats["n_encoded"],
        "save_dir": str(save_dir),
        "output": str(out_path),
        "environment_note": "Use the maintained pinned upstream revision; model/provider identity is configured at runtime.",
    }
    manifest["wall_seconds"] = time.perf_counter() - run_started
    rss_delta = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss_started
    # macOS reports bytes; Linux reports KiB. Preserve the platform-correct
    # unit rather than publishing a misleading cross-platform field name.
    manifest["max_rss_delta"] = rss_delta
    manifest["max_rss_unit"] = "bytes" if platform.system() == "Darwin" else "KiB"
    out_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    # Running this file directly places candidate_pool/ rather than the repository root
    # on sys.path. Keep the historical direct CLI working while still reading
    # every new parameter from the canonical config module.
    try:
        import configuration as project_config
    except ModuleNotFoundError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import configuration as project_config

    entry_cfg = project_config.params("hipporag_entry", default={})
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--save-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-docs", type=int, required=True)
    ap.add_argument("--max-queries", type=int, required=True)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--openie-workers", type=int, default=2)
    ap.add_argument(
        "--retrieval-profile",
        choices=RETRIEVAL_PROFILES,
        required=True,
        help="Retrieval-only ablation; all profiles reuse the same indexed graph.",
    )
    ap.add_argument(
        "--llm-model",
        required=True,
    )
    ap.add_argument("--embedding-model", required=True)
    ap.add_argument(
        "--normalize-transformer-embeddings",
        action="store_true",
        help=(
            "L2-normalize local Transformers embeddings so HippoRAG dot "
            "products match the frozen SBERT cosine baseline."
        ),
    )
    ap.add_argument(
        "--e5-role-prefixes",
        action="store_true",
        help=(
            "Apply canonical E5 retrieval prefixes: query: for HippoRAG "
            "query-instruction calls and passage: for indexed inputs."
        ),
    )
    ap.add_argument("--entry-trace", action="store_true",
                    help="Write query→fact/passsage entry provenance; does not change ranking.")
    ap.add_argument(
        "--prepend-title",
        action="store_true",
        help="Build title+'\\n'+text documents exactly like upstream main.py.",
    )
    ap.add_argument(
        "--openie-cache-provenance",
        help="Actual extractor model when a precomputed cache is deliberately reused.",
    )
    ap.add_argument("--entry-trace-top-k", type=int, default=int(
        entry_cfg.get("trace_top_k", 20)))
    ap.add_argument(
        "--test-split-used", action="store_true",
        help=("Declare a frozen test adapter. Requires exact SHA-256 and row "
              "count; the default validation-only behaviour is unchanged."),
    )
    ap.add_argument("--expected-query-sha256")
    ap.add_argument("--expected-query-count", type=int)
    ap.add_argument(
        "--forbid-llm-inference", action="store_true",
        help=("Fail before any LLM/OpenIE inference call; use only with a "
              "complete frozen cache and recognition-free retrieval."),
    )
    args = ap.parse_args()
    print(json.dumps(run(
        corpus_path=args.corpus,
        queries_path=args.queries,
        save_dir=args.save_dir,
        out_path=args.out,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        openie_workers=args.openie_workers,
        normalize_transformer_embeddings=args.normalize_transformer_embeddings,
        e5_role_prefixes=args.e5_role_prefixes,
        retrieval_profile=args.retrieval_profile,
        entry_trace=args.entry_trace,
        entry_trace_top_k=args.entry_trace_top_k,
        prepend_title=args.prepend_title,
        openie_cache_provenance=args.openie_cache_provenance,
        test_split_used=args.test_split_used,
        expected_query_sha256=args.expected_query_sha256,
        expected_query_count=args.expected_query_count,
        forbid_llm_inference=args.forbid_llm_inference,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
