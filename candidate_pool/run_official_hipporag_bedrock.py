#!/usr/bin/env python3
"""Run the maintained HippoRAG implementation on the validation adapter.

This runner deliberately accepts only the validation adapter emitted by
``export_hipporag_dataset.py``.  It rebuilds HippoRAG's OpenIE graph rather
than reusing this project's graph, so its output is an *official end-to-end*
baseline, distinct from ``MultiHopRetriever.retrieve_ppr``.

Use a dedicated ``conda`` environment containing HippoRAG's upstream main
branch.  The model prefix is LiteLLM's syntax; Anthropic 4.5 Haiku needs the
Bedrock global inference profile in this account.
"""
from __future__ import annotations

import argparse
import ast
import csv
import functools
import json
import math
import os
import platform
import resource
import time
from datetime import datetime, timezone
from hashlib import md5, sha256
from pathlib import Path
from types import MethodType


RETRIEVAL_PROFILES = (
    "official",
    "dense_only",
    "no_recognition",
    "fact_only_no_recognition",
    "equal_fact_no_recognition",
    "temperature_fact_no_recognition",
    "ontology_bridge_no_recognition",
    "ontology_seed_bridge_no_recognition",
    "soft_recognition",
    "adhd_need_aware",
    "no_dense_teleport",
)

_SECTION17_COMPONENTS = (
    "primary_problem", "constraints", "failed_attempts", "desired_outcome",
    "additional_need",
)


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


def _seed_mixture_components(decomposition: dict, weights: dict[str, float]) -> list[tuple[str, float, str]]:
    """Return (text, normalized weight, facet) without reading any labels."""
    components = []
    for field in _SECTION17_COMPONENTS:
        value = decomposition.get(field)
        values = value if isinstance(value, list) else ([value] if value else [])
        values = [str(x).strip() for x in values if str(x).strip()]
        field_weight = float(weights.get(field, 1.0))
        for text in values:
            components.append((text, field_weight / max(len(values), 1), field))
    total = sum(x[1] for x in components) or 1.0
    return [(text, weight / total, field) for text, weight, field in components]


def _install_multiquery_seed_mixture(
    hipporag, components_by_query: dict[str, list[tuple[str, float, str]]],
) -> dict:
    """Mix component fact and passage score vectors before official PPR.

    Recognition, specificity, graph transition, PPR and passage aggregation
    remain upstream.  Only the two query-conditioned reset-score sources are
    replaced by a fixed weighted mixture of scores from the query facets.
    """
    import numpy as np
    original_fact = hipporag.get_fact_scores
    original_dense = hipporag.dense_passage_retrieval
    runtime = {"queries": 0, "fallback_queries": 0, "component_calls": 0}

    def _fact_scores(self, query):
        parts = components_by_query.get(query, [])
        if not parts:
            runtime["fallback_queries"] += 1
            return original_fact(query)
        # Upstream's individual-score fallback passes a bare string to
        # batch_encode, which some backends interpret as a character batch.
        # Follow the official lifecycle explicitly for new component strings.
        if hasattr(self, "get_query_embeddings"):
            self.get_query_embeddings([text for text, _, _ in parts])
        mixed = None
        for text, weight, _ in parts:
            values = np.asarray(original_fact(text), dtype=float)
            mixed = weight * values if mixed is None else mixed + weight * values
            runtime["component_calls"] += 1
        runtime["queries"] += 1
        return mixed

    def _dense_scores(self, query):
        parts = components_by_query.get(query, [])
        if not parts:
            return original_dense(query)
        n = len(self.passage_node_keys)
        mixed = np.zeros(n, dtype=float)
        for text, weight, _ in parts:
            ids, scores = original_dense(text)
            mixed[np.asarray(ids, dtype=int)] += weight * np.asarray(scores, dtype=float)
        order = np.argsort(-mixed, kind="stable")
        return order, mixed[order]

    hipporag.get_fact_scores = MethodType(_fact_scores, hipporag)
    hipporag.dense_passage_retrieval = MethodType(_dense_scores, hipporag)
    return {
        "enabled": True, "level": "pre-PPR fact+passage reset scores",
        "graph_construction_changed": False, "recognition_changed": False,
        "ppr_changed": False, "runtime": runtime,
    }


def _load_agreed_evidence_types(path: Path) -> tuple[dict[str, str], dict]:
    """Load only query types agreed across all available blind passes."""
    by_query: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            qid = str(row.get("query_id") or "").strip()
            qtype = str(row.get("primary_type") or "").strip()
            if qid and qtype:
                by_query.setdefault(qid, []).append(qtype)
    agreed = {
        qid: types[0] for qid, types in by_query.items()
        if types and len(set(types)) == 1
    }
    return agreed, {
        "labelled_queries": len(by_query),
        "agreed_queries": len(agreed),
        "disagreed_queries": len(by_query) - len(agreed),
        "source": str(path),
    }


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


def _reweight_selected_fact_scores(
    query_fact_scores,
    selected_indices: list[int],
    mode: str,
    temperature: float,
) -> list[float]:
    """Mutate only selected fact scores and return their resulting weights.

    HippoRAG's downstream graph search reads the same score array after fact
    selection.  Keeping identities fixed while changing these values isolates
    seed weighting from graph construction, dense teleport and PPR.
    """
    if mode == "similarity":
        return [float(query_fact_scores[idx]) for idx in selected_indices]
    if mode == "equal":
        for idx in selected_indices:
            query_fact_scores[idx] = 1.0
        return [1.0] * len(selected_indices)
    if mode != "temperature":
        raise ValueError(f"unknown fact seed weighting: {mode}")
    if temperature <= 0:
        raise ValueError("fact seed temperature must be positive")
    if not selected_indices:
        return []
    values = [float(query_fact_scores[idx]) / temperature for idx in selected_indices]
    offset = max(values)
    exp_values = [math.exp(value - offset) for value in values]
    denom = sum(exp_values) or 1.0
    weights = [value / denom for value in exp_values]
    for idx, weight in zip(selected_indices, weights, strict=True):
        query_fact_scores[idx] = weight
    return weights


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


def _install_retrieval_profile(
    hipporag,
    profile: str,
    *,
    soft_rejected_floor: float = 0.25,
    query_types: dict[str, str] | None = None,
    need_aware_floors: dict[str, float] | None = None,
    default_evidence_type: str = "direct_match",
    recognition_trace: list[dict] | None = None,
    fact_seed_temperature: float = 0.10,
) -> dict:
    """Apply one retrieval-only ablation without changing the indexed graph.

    The full official profile is untouched.  ``no_recognition`` preserves the
    upstream query-to-fact embedding scores and top-k contract but bypasses the
    DSPy/LLM recognition filter. ``soft_recognition`` runs the upstream filter
    but retains rejected top-k facts with reduced seed mass. ``adhd_need_aware``
    selects that residual mass from blind query-level evidence-need labels.
    ``no_dense_teleport`` preserves official recognition while setting passage
    reset mass to zero. The three ``*_no_recognition`` entry profiles keep the
    same embedding top-k fact identities, then isolate passage teleport and
    fact-score calibration. Dense-only uses upstream dense retrieval.
    """
    if profile not in RETRIEVAL_PROFILES:
        raise ValueError(f"unknown retrieval profile: {profile}")

    if not 0.0 <= soft_rejected_floor <= 1.0:
        raise ValueError("soft_rejected_floor must be in [0, 1]")
    query_types = query_types or {}
    need_aware_floors = need_aware_floors or {}
    for name, floor in need_aware_floors.items():
        if not 0.0 <= float(floor) <= 1.0:
            raise ValueError(f"need-aware floor for {name!r} must be in [0, 1]")

    no_recognition_profiles = {
        "no_recognition", "fact_only_no_recognition",
        "equal_fact_no_recognition", "temperature_fact_no_recognition",
        "ontology_bridge_no_recognition", "ontology_seed_bridge_no_recognition",
    }
    changes = {
        "profile": profile,
        "graph_construction_changed": profile in {
            "ontology_bridge_no_recognition",
            "ontology_seed_bridge_no_recognition",
        },
        "recognition_filter_enabled": profile not in no_recognition_profiles,
        "recognition_filter_mode": {
            "no_recognition": "bypassed",
            "fact_only_no_recognition": "bypassed",
            "equal_fact_no_recognition": "bypassed",
            "temperature_fact_no_recognition": "bypassed",
            "ontology_bridge_no_recognition": "bypassed",
            "ontology_seed_bridge_no_recognition": "bypassed",
            "soft_recognition": "upstream_hard_plus_soft_rejected_facts",
            "adhd_need_aware": "upstream_hard_plus_need_conditioned_soft_rejected_facts",
        }.get(profile, "upstream_hard"),
        "dense_passage_teleport_enabled": profile not in {
            "no_dense_teleport", "fact_only_no_recognition"},
        "ppr_enabled": profile != "dense_only",
    }
    if profile in {"no_dense_teleport", "fact_only_no_recognition"}:
        hipporag.global_config.passage_node_weight = 0.0
    if profile in no_recognition_profiles:
        weighting = {
            "equal_fact_no_recognition": "equal",
            "temperature_fact_no_recognition": "temperature",
        }.get(profile, "similarity")
        changes["fact_seed_weighting"] = weighting
        if weighting == "temperature":
            changes["fact_seed_temperature"] = fact_seed_temperature

        def _bypass_recognition(self, query, query_fact_scores):
            del query  # ranking is already query-conditioned by embedding scores
            indices, facts = _top_embedding_facts(self, query_fact_scores)
            resulting_weights = _reweight_selected_fact_scores(
                query_fact_scores, indices, weighting, fact_seed_temperature)
            return indices, facts, {
                "facts_before_rerank": facts,
                "facts_after_rerank": facts,
                "ablation": "recognition_filter_bypassed",
                "fact_seed_weighting": weighting,
                "selected_fact_weights": resulting_weights,
            }

        hipporag.rerank_facts = MethodType(_bypass_recognition, hipporag)
    elif profile in {"soft_recognition", "adhd_need_aware"}:
        upstream_rerank = hipporag.rerank_facts
        runtime = {
            "queries": 0,
            "upstream_selected_facts": 0,
            "soft_retained_rejected_facts": 0,
            "evidence_type_counts": {},
        }
        changes["soft_rejected_floor"] = soft_rejected_floor
        changes["need_aware_floors"] = need_aware_floors
        changes["default_evidence_type"] = default_evidence_type
        changes["runtime"] = runtime

        def _soft_recognition(self, query, query_fact_scores):
            selected_indices, selected_facts, upstream_log = upstream_rerank(
                query, query_fact_scores)
            candidate_indices, candidate_facts = _top_embedding_facts(
                self, query_fact_scores)
            if not candidate_indices:
                return selected_indices, selected_facts, upstream_log

            evidence_type = query_types.get(query, default_evidence_type)
            floor = soft_rejected_floor
            if profile == "adhd_need_aware":
                floor = float(need_aware_floors.get(
                    evidence_type,
                    need_aware_floors.get(default_evidence_type, 0.0),
                ))
            runtime["queries"] += 1
            runtime["upstream_selected_facts"] += len(selected_indices)
            runtime["evidence_type_counts"][evidence_type] = (
                runtime["evidence_type_counts"].get(evidence_type, 0) + 1)

            selected = set(selected_indices)
            candidate_scores = [float(query_fact_scores[idx]) for idx in candidate_indices]
            # A zero floor must reproduce the upstream hard-filter branch,
            # including its dense fallback when recognition keeps no facts.
            if floor <= 0.0:
                if recognition_trace is not None:
                    recognition_trace.append({
                        "query_text": query,
                        "evidence_type": evidence_type,
                        "rejected_floor": 0.0,
                        "candidate_facts": [list(fact) for fact in candidate_facts],
                        "candidate_embedding_scores": candidate_scores,
                        "upstream_selected_positions": [
                            pos for pos, idx in enumerate(candidate_indices) if idx in selected
                        ],
                        "upstream_selected_facts": [list(fact) for fact in selected_facts],
                        "rejected_positions_soft_retained": [],
                    })
                return selected_indices, selected_facts, {
                    **upstream_log,
                    "adaptation": "hard_recognition_preserved",
                    "evidence_type": evidence_type,
                    "rejected_floor": 0.0,
                }

            rejected = [idx for idx in candidate_indices if idx not in selected]
            if recognition_trace is not None:
                recognition_trace.append({
                    "query_text": query,
                    "evidence_type": evidence_type,
                    "rejected_floor": floor,
                    "candidate_facts": [list(fact) for fact in candidate_facts],
                    "candidate_embedding_scores": candidate_scores,
                    "upstream_selected_positions": [
                        pos for pos, idx in enumerate(candidate_indices) if idx in selected
                    ],
                    "upstream_selected_facts": [list(fact) for fact in selected_facts],
                    "rejected_positions_soft_retained": [
                        pos for pos, idx in enumerate(candidate_indices) if idx not in selected
                    ],
                })
            for idx in rejected:
                query_fact_scores[idx] = max(0.0, float(query_fact_scores[idx])) * floor
            runtime["soft_retained_rejected_facts"] += len(rejected)
            return candidate_indices, candidate_facts, {
                **upstream_log,
                "facts_after_softening": candidate_facts,
                "adaptation": "soft_rejected_fact_retention",
                "evidence_type": evidence_type,
                "rejected_floor": floor,
                "upstream_selected_indices": selected_indices,
            }

        hipporag.rerank_facts = MethodType(_soft_recognition, hipporag)
    return changes


def _retrieve_dense_only(hipporag, query: str, top_k: int):
    """Return an upstream QuerySolution-shaped dense-only result."""
    import numpy as np

    sorted_ids, sorted_scores = hipporag.dense_passage_retrieval(query)
    ids = sorted_ids[:top_k].tolist()
    docs = [
        hipporag.chunk_embedding_store.get_row(hipporag.passage_node_keys[idx])["content"]
        for idx in ids
    ]
    # The runner only relies on these two attributes, so a tiny local carrier
    # avoids copying or depending on HippoRAG's internal QuerySolution class.
    return type("DenseSolution", (), {
        "docs": docs,
        "doc_scores": np.asarray(sorted_scores[:top_k]),
    })()


def _prepare_relation_assets(hipporag, cache_dir: Path, save_dir: Path):
    """Load-or-build the relation sidecar + label embeddings (shared cache).

    Single source of truth for every stage that needs relation compatibility
    (transition ``node_relation``, spreading ``spread_node_relation_2hop``,
    ``dense_bridge_relation``).  Never mutates the graph.
    """
    from candidate_pool.retrieval.relation_sidecar import (
        build_relation_sidecar, load_or_encode_relation_embeddings,
        load_relation_sidecar, save_relation_sidecar,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = cache_dir / "relation_sidecar.npz"
    if sidecar_path.exists():
        sidecar = load_relation_sidecar(sidecar_path)
        if sidecar.graph_edges != hipporag.graph.ecount():
            raise RuntimeError("cached relation sidecar graph edge count mismatch")
    else:
        openie_path = Path(hipporag.openie_results_path)
        if not openie_path.exists():
            candidates = sorted(save_dir.glob("openie_results*.json"))
            if not candidates:
                raise FileNotFoundError("saved Official OpenIE cache not found")
            openie_path = candidates[0]
        sidecar = build_relation_sidecar(
            hipporag.graph, json.loads(openie_path.read_text(encoding="utf-8")))
        save_relation_sidecar(sidecar, sidecar_path)
    from hipporag.prompts.linking import get_query_instruction
    relation_embeddings, manifest = load_or_encode_relation_embeddings(
        sidecar, hipporag.embedding_model,
        cache_dir / "relation_embeddings.npy",
        instruction=get_query_instruction("query_to_fact"))
    return sidecar, relation_embeddings, manifest


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
    retrieval_profile: str = "official",
    evidence_requirements: Path | None = None,
    only_labelled_queries: bool = False,
    soft_rejected_floor: float = 0.25,
    need_aware_floors: dict[str, float] | None = None,
    default_evidence_type: str = "direct_match",
    fact_seed_temperature: float = 0.10,
    entry_trace: bool = False,
    entry_trace_top_k: int = 20,
    ontology_spec: Path | None = None,
    ontology_top_k_per_concept: int = 12,
    ontology_min_similarity: float = 0.35,
    ontology_max_concepts_per_entity: int = 2,
    ontology_bridge_weight: float = 0.20,
    ontology_hierarchy_weight: float = 0.05,
    ontology_seed_weight: float = 0.10,
    ontology_seed_top_k: int = 3,
    prepend_title: bool = False,
    openie_cache_provenance: str | None = None,
    seed_mixture_decompositions: Path | None = None,
    seed_mixture_weights: dict[str, float] | None = None,
    transition_profiles: list[str] | None = None,
    transition_cache_dir: Path | None = None,
    transition_alpha: float = 1.0,
    transition_beta: float = 1.0,
    transition_hub_gamma: float = 0.5,
    transition_epsilon: float = 0.05,
    transition_relation_aggregation: str = "max",
    local_pcst: bool = False,
    local_pcst_params: dict | None = None,
    ircot: bool = False,
    ircot_params: dict | None = None,
    ircot_cache_dir: Path | None = None,
    dense_bridge: bool = False,
    dense_bridge_params: dict | None = None,
    spreading_variants: list[str] | None = None,
    spreading_params: dict | None = None,
    multi_pcst_variants: list[str] | None = None,
    multi_pcst_params: dict | None = None,
    stage_cache_dir: Path | None = None,
    test_split_used: bool = False,
    expected_query_sha256: str | None = None,
    expected_query_count: int | None = None,
    forbid_llm_inference: bool = False,
) -> dict:
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
    evidence_types: dict[str, str] = {}
    evidence_manifest: dict = {"source": None, "agreed_queries": 0}
    if evidence_requirements is not None:
        evidence_types, evidence_manifest = _load_agreed_evidence_types(
            evidence_requirements)
        if only_labelled_queries:
            queries = [q for q in queries if _adapter_query_id(q) in evidence_types]
    elif only_labelled_queries:
        raise ValueError("--only-labelled-queries requires --evidence-requirements")
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
    ontology_manifest: dict = {"enabled": False}
    ontology_trace_by_query: dict[str, dict] = {}
    ontology_profiles = {
        "ontology_bridge_no_recognition",
        "ontology_seed_bridge_no_recognition",
    }
    if retrieval_profile in ontology_profiles:
        if ontology_spec is None:
            raise ValueError(
                f"{retrieval_profile} requires --ontology-spec")
        # The bridge consumes the exact upstream entity keys/embeddings and
        # node mapping, which are initialized by this canonical lifecycle call.
        # It mutates only this in-memory Graph object and never save_igraph().
        if not hipporag.ready_to_retrieve:
            hipporag.prepare_retrieval_objects()
        from candidate_pool.retrieval.ontology_hipporag_bridge import (
            augment_hipporag_with_ontology,
        )
        effective_seed_weight = (
            ontology_seed_weight
            if retrieval_profile == "ontology_seed_bridge_no_recognition"
            else 0.0
        )
        ontology_manifest = augment_hipporag_with_ontology(
            hipporag,
            ontology_spec,
            top_k_per_concept=ontology_top_k_per_concept,
            min_similarity=ontology_min_similarity,
            max_concepts_per_entity=ontology_max_concepts_per_entity,
            bridge_weight=ontology_bridge_weight,
            hierarchy_weight=ontology_hierarchy_weight,
            theory_seed_weight=effective_seed_weight,
            theory_seed_top_k=ontology_seed_top_k,
            trace_sink=ontology_trace_by_query,
        )
    transition_profiles = list(dict.fromkeys(transition_profiles or []))
    transition_sink: dict[str, dict] = {}
    transition_runtime = None
    transition_adapter = None
    transition_sidecar_manifest = None
    transition_relation_embedding_manifest = None
    transition_state_before = None
    if transition_profiles:
        if retrieval_profile != "official":
            raise ValueError("transition profiles require --retrieval-profile official")
        if seed_mixture_decompositions is not None:
            raise ValueError("transition profiles isolate transitions and cannot mix Section17 seeds")
        if not hipporag.ready_to_retrieve:
            hipporag.prepare_retrieval_objects()
        from candidate_pool.retrieval.official_graph_adapter import OfficialGraphAdapter
        from candidate_pool.retrieval.official_query_aware_ppr import (
            TransitionParameters, install_transition_profiles,
        )
        cache_dir = transition_cache_dir or (out_path.parent / "transition_cache")
        needs_relation = any("relation" in profile for profile in transition_profiles)
        sidecar = None
        relation_embeddings = None
        if needs_relation:
            sidecar, relation_embeddings, transition_relation_embedding_manifest = (
                _prepare_relation_assets(hipporag, cache_dir, save_dir))
            transition_sidecar_manifest = sidecar.coverage
        transition_adapter = OfficialGraphAdapter(
            hipporag, Path(hipporag._graph_pickle_filename),
            relation_sidecar=sidecar, relation_embeddings=relation_embeddings)
        transition_state_before = transition_adapter.state_signature()
        transition_runtime = install_transition_profiles(
            hipporag, transition_adapter, transition_profiles, transition_sink,
            top_k=top_k,
            params=TransitionParameters(
                alpha=transition_alpha, beta=transition_beta,
                hub_gamma=transition_hub_gamma, epsilon=transition_epsilon,
                relation_aggregation=transition_relation_aggregation))
    pcst_sink: dict[str, dict] = {}
    pcst_runtime = None
    pcst_adapter = None
    pcst_state_before = None
    if local_pcst:
        if transition_profiles:
            raise ValueError("run local PCST separately from transition profiles")
        if retrieval_profile != "official" or seed_mixture_decompositions is not None:
            raise ValueError("local PCST requires the unmodified official retrieval lifecycle")
        if not hipporag.ready_to_retrieve:
            hipporag.prepare_retrieval_objects()
        from candidate_pool.retrieval.official_graph_adapter import OfficialGraphAdapter
        from candidate_pool.retrieval.local_pcst_retriever import (
            LocalPCSTParameters, LocalPCSTRetriever, install_local_pcst,
        )
        pcst_adapter = OfficialGraphAdapter(
            hipporag, Path(hipporag._graph_pickle_filename))
        pcst_state_before = pcst_adapter.state_signature()
        params = LocalPCSTParameters(**dict(local_pcst_params or {}))
        pcst_runtime = install_local_pcst(
            hipporag, LocalPCSTRetriever(pcst_adapter, params), pcst_sink,
            top_k=top_k)
    if ircot:
        # One-step IRCoT is a post-retrieval stage: both rounds must run the
        # unmodified official lifecycle, so it cannot be combined with any
        # transition/PCST/seed-mixture modification.
        if transition_profiles or local_pcst:
            raise ValueError("run one-step IRCoT separately from transition/PCST stages")
        if retrieval_profile != "official" or seed_mixture_decompositions is not None:
            raise ValueError("one-step IRCoT requires the unmodified official retrieval lifecycle")
    # ---- Round-2 stages: dense bridge / spreading activation / multi-PCST.
    # Each stage is run in isolation (one modification per run) on the
    # unmodified official lifecycle, exactly like transition/PCST/IRCoT.
    round2_requested = [name for name, enabled in [
        ("dense_bridge", dense_bridge),
        ("spreading", bool(spreading_variants)),
        ("multi_pcst", bool(multi_pcst_variants)),
    ] if enabled]
    if round2_requested:
        if len(round2_requested) > 1 or transition_profiles or local_pcst or ircot:
            raise ValueError(
                f"run {round2_requested} in separate isolated runs "
                "(one retrieval modification per run)")
        if retrieval_profile != "official" or seed_mixture_decompositions is not None:
            raise ValueError(
                f"{round2_requested[0]} requires the unmodified official retrieval lifecycle")
        if not hipporag.ready_to_retrieve:
            hipporag.prepare_retrieval_objects()

    def _round2_adapter(needs_relation: bool):
        """Adapter (+ shared sidecar assets) for every round-2 stage."""
        from candidate_pool.retrieval.official_graph_adapter import OfficialGraphAdapter
        sidecar = None
        relation_embeddings = None
        relation_manifest = None
        if needs_relation:
            cache_dir = stage_cache_dir or (out_path.parent / "transition_cache")
            sidecar, relation_embeddings, relation_manifest = (
                _prepare_relation_assets(hipporag, cache_dir, save_dir))
        adapter = OfficialGraphAdapter(
            hipporag, Path(hipporag._graph_pickle_filename),
            relation_sidecar=sidecar, relation_embeddings=relation_embeddings)
        return adapter, relation_manifest

    spreading_sink: dict[str, dict] = {}
    spreading_runtime = None
    spreading_adapter = None
    spreading_state_before = None
    spreading_params_obj = None
    if spreading_variants:
        from candidate_pool.retrieval.spreading_activation import (
            SpreadingActivation, SpreadingParameters, install_spreading,
        )
        spreading_params_obj = SpreadingParameters(**dict(spreading_params or {}))
        needs_relation = any("relation" in v for v in spreading_variants)
        spreading_adapter, _ = _round2_adapter(needs_relation)
        spreading_state_before = spreading_adapter.state_signature()
        spreading_runtime = install_spreading(
            hipporag, SpreadingActivation(spreading_adapter, spreading_params_obj),
            list(spreading_variants), spreading_sink, top_k=top_k)
    multi_pcst_sink: dict[str, dict] = {}
    multi_pcst_runtime = None
    multi_pcst_adapter = None
    multi_pcst_state_before = None
    if multi_pcst_variants:
        from candidate_pool.retrieval.local_pcst_retriever import (
            LocalPCSTParameters, LocalPCSTRetriever,
        )
        from candidate_pool.retrieval.multi_pcst import (
            MultiPCSTParameters, MultiPCSTRetriever, install_multi_pcst,
        )
        multi_pcst_adapter, _ = _round2_adapter(needs_relation=False)
        multi_pcst_state_before = multi_pcst_adapter.state_signature()
        local = LocalPCSTRetriever(
            multi_pcst_adapter, LocalPCSTParameters(**dict(local_pcst_params or {})))
        multi_params_obj = MultiPCSTParameters(**dict(multi_pcst_params or {}))
        multi_pcst_runtime = install_multi_pcst(
            hipporag,
            {variant: MultiPCSTRetriever(local, multi_params_obj, variant)
             for variant in multi_pcst_variants},
            multi_pcst_sink, top_k=top_k)
    bridge_entered: set[str] = set()
    bridge_adapter = None
    bridge_state_before = None
    if dense_bridge:
        from candidate_pool.retrieval.dense_graph_bridge import install_graph_entry_marker
        needs_relation = True  # dense_bridge_relation is part of the variant set
        bridge_adapter, _ = _round2_adapter(needs_relation)
        bridge_state_before = bridge_adapter.state_signature()
        install_graph_entry_marker(hipporag, bridge_entered)
    query_types_by_text = {
        str(q["question"]): evidence_types.get(
            _adapter_query_id(q), default_evidence_type)
        for q in queries
    }
    recognition_trace: list[dict] = []
    profile_manifest = _install_retrieval_profile(
        hipporag,
        retrieval_profile,
        soft_rejected_floor=soft_rejected_floor,
        query_types=query_types_by_text,
        need_aware_floors=need_aware_floors,
        default_evidence_type=default_evidence_type,
        recognition_trace=recognition_trace,
        fact_seed_temperature=fact_seed_temperature,
    )
    seed_mixture_manifest: dict = {"enabled": False}
    if seed_mixture_decompositions is not None:
        if retrieval_profile != "official":
            raise ValueError("section17 seed mixture currently requires --retrieval-profile official")
        if "test" in seed_mixture_decompositions.name.lower():
            raise ValueError("test decomposition is frozen")
        decomposition_rows = [json.loads(line) for line in
                              seed_mixture_decompositions.read_text(encoding="utf-8").splitlines()
                              if line.strip()]
        decomposition_by_id = {str(row["query_id"]): row for row in decomposition_rows}
        weights = dict(seed_mixture_weights or {})
        components_by_text = {}
        component_summary = {}
        for query in queries:
            qid, qtext = _adapter_query_id(query), str(query["question"])
            parts = _seed_mixture_components(decomposition_by_id.get(qid, {}), weights)
            components_by_text[qtext] = parts
            component_summary[qid] = [
                {"component": field, "weight": weight, "text": text}
                for text, weight, field in parts
            ]
        seed_mixture_manifest = _install_multiquery_seed_mixture(
            hipporag, components_by_text)
        seed_mixture_manifest.update({
            "source": str(seed_mixture_decompositions), "weights": weights,
            "components_by_query": component_summary,
        })
    entry_trace_by_query: dict[str, dict] = {}
    if entry_trace:
        _install_entry_trace(
            hipporag, entry_trace_by_query, top_k=entry_trace_top_k)
    query_texts = [str(row["question"]) for row in queries]
    if retrieval_profile == "dense_only":
        # These are the same lifecycle calls made at the start of upstream
        # ``retrieve``.  Dense-only deliberately bypasses the later fact/PPR
        # loop, not preparation or the instructed query embedding path.
        if not hipporag.ready_to_retrieve:
            hipporag.prepare_retrieval_objects()
        hipporag.get_query_embeddings(query_texts)
        solutions = [_retrieve_dense_only(hipporag, query, top_k) for query in query_texts]
    else:
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
    transition_outputs = {}
    transition_trace_path = out_path.with_suffix(".transition_trace.jsonl")
    if transition_profiles:
        transition_runtime["retrieval_queries"] = len(queries)
        transition_runtime["dense_fallback_queries"] = sum(
            str(query["question"]) not in transition_sink for query in queries)
        with transition_trace_path.open("w", encoding="utf-8") as trace_fh:
            for query, official_solution in zip(queries, solutions, strict=True):
                qid, query_text = _adapter_query_id(query), str(query["question"])
                profile_payload = transition_sink.get(query_text, {})
                trace_fh.write(json.dumps({
                    "query_id": qid, "query_text": query_text,
                    "profiles": {name: {
                        key: value for key, value in payload.items()
                        if key not in {"doc_ids", "doc_scores"}
                    } for name, payload in profile_payload.items()},
                    "dense_fallback": not bool(profile_payload),
                }, ensure_ascii=False) + "\n")
        for profile in transition_profiles:
            profile_path = out_path.with_name(
                f"{out_path.stem}.transition_{profile}{out_path.suffix}")
            profile_rows = []
            for query, official_solution in zip(queries, solutions, strict=True):
                query_text = str(query["question"])
                payload = transition_sink.get(query_text, {}).get(profile)
                if payload is None:
                    docs_list = list(official_solution.docs)
                    scores_list = [float(x) for x in official_solution.doc_scores]
                else:
                    ids = payload["doc_ids"].tolist()
                    docs_list = [hipporag.chunk_embedding_store.get_row(
                        hipporag.passage_node_keys[idx])["content"] for idx in ids]
                    scores_list = [float(x) for x in payload["doc_scores"]]
                profile_rows.append({
                    "query_id": _adapter_query_id(query),
                    "query_text": query_text,
                    "retrieved_titles": [text_to_title.get(str(t), "") for t in docs_list],
                    "retrieved_texts": docs_list,
                    "retrieved_scores": scores_list,
                    "gold_titles": _gold_titles(query),
                    "gold_texts": _optional_gold_texts(query),
                })
            with profile_path.open("w", encoding="utf-8") as fh:
                for row in profile_rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            transition_outputs[profile] = str(profile_path)
    pcst_output_path = out_path.with_name(f"{out_path.stem}.local_pcst{out_path.suffix}")
    pcst_trace_path = out_path.with_suffix(".local_pcst_trace.jsonl")
    if local_pcst:
        pcst_runtime["retrieval_queries"] = len(queries)
        pcst_runtime["dense_fallback_queries"] = sum(
            str(query["question"]) not in pcst_sink for query in queries)
        with pcst_output_path.open("w", encoding="utf-8") as output_fh, \
                pcst_trace_path.open("w", encoding="utf-8") as trace_fh:
            for query, official_solution in zip(queries, solutions, strict=True):
                qid, query_text = _adapter_query_id(query), str(query["question"])
                payload = pcst_sink.get(query_text)
                if payload is None:
                    docs_list = list(official_solution.docs)
                    scores_list = [float(x) for x in official_solution.doc_scores]
                    diagnostics = {"fallback": "official_dense_fallback"}
                else:
                    ids = payload["doc_ids"].tolist()
                    docs_list = [hipporag.chunk_embedding_store.get_row(
                        hipporag.passage_node_keys[idx])["content"] for idx in ids]
                    scores_list = [float(x) for x in payload["doc_scores"]]
                    diagnostics = payload["diagnostics"]
                output_fh.write(json.dumps({
                    "query_id": qid, "query_text": query_text,
                    "retrieved_titles": [text_to_title.get(str(t), "") for t in docs_list],
                    "retrieved_texts": docs_list, "retrieved_scores": scores_list,
                    "gold_titles": _gold_titles(query), "gold_texts": _optional_gold_texts(query),
                }, ensure_ascii=False) + "\n")
                trace_fh.write(json.dumps({
                    "query_id": qid, "query_text": query_text,
                    "diagnostics": diagnostics,
                }, ensure_ascii=False) + "\n")
    ircot_output_path = out_path.with_name(f"{out_path.stem}.ircot{out_path.suffix}")
    ircot_trace_path = out_path.with_suffix(".ircot_trace.jsonl")
    ircot_stats = None
    ircot_params_obj = None
    ircot_graph_sha = None
    ircot_cache = None
    if ircot:
        from candidate_pool.retrieval.iterative_retrieval import (
            GenerationCache, IRCoTParameters, run_one_step_ircot,
        )
        from candidate_pool.retrieval.official_graph_adapter import file_sha256
        from shared.llm_client import call_chat
        graph_pickle_path = Path(hipporag._graph_pickle_filename)
        graph_sha_before = file_sha256(graph_pickle_path)
        ircot_params_obj = IRCoTParameters(**dict(ircot_params or {}))
        title_to_text = {title: text for text, title in text_to_title.items()}
        first_rankings: dict[str, list[str]] = {}
        first_titles_texts: dict[str, tuple[list[str], list[str]]] = {}
        ircot_queries = []
        for query, solution in zip(queries, solutions, strict=True):
            qid = _adapter_query_id(query)
            docs_list = [str(t) for t in solution.docs]
            titles = [text_to_title.get(t, "") for t in docs_list]
            first_rankings[qid] = titles
            first_titles_texts[qid] = (titles, docs_list)
            ircot_queries.append(
                {"query_id": qid, "question": str(query["question"])})
        cache_dir = ircot_cache_dir or (out_path.parent / "ircot_cache")
        # call_chat uses ``provider:model``; the runner historically uses the
        # LiteLLM-style ``bedrock/model`` spec.
        chat_spec = (llm_model.replace("/", ":", 1)
                     if llm_model.startswith("bedrock/") else llm_model)
        ircot_cache = GenerationCache(cache_dir / "cot_generations.json", chat_spec)

        def _ircot_generate(prompt: str) -> str:
            return call_chat(prompt, chat_spec,
                             max_tokens=ircot_params_obj.llm_max_tokens,
                             temperature=ircot_params_obj.llm_temperature)

        def _ircot_retrieve_second(second_queries: list[str]) -> dict[str, list[str]]:
            # SAME retriever instance, profile hooks and top_k as round one.
            second_solutions = hipporag.retrieve(
                second_queries, num_to_retrieve=top_k)
            return {
                second_query: [text_to_title.get(str(t), "") for t in sol.docs]
                for second_query, sol in zip(
                    second_queries, second_solutions, strict=True)}

        ircot_rankings, ircot_trace_rows, ircot_stats = run_one_step_ircot(
            ircot_queries, first_rankings, first_titles_texts,
            params=ircot_params_obj, top_k=top_k,
            generate_fn=_ircot_generate,
            retrieve_second_fn=_ircot_retrieve_second, cache=ircot_cache)
        ircot_graph_sha = {
            "before": graph_sha_before,
            "after": file_sha256(graph_pickle_path),
        }
        with ircot_output_path.open("w", encoding="utf-8") as output_fh, \
                ircot_trace_path.open("w", encoding="utf-8") as trace_fh:
            for query, trace_row in zip(queries, ircot_trace_rows, strict=True):
                qid = _adapter_query_id(query)
                merged_ids = ircot_rankings[qid]
                output_fh.write(json.dumps({
                    "query_id": qid,
                    "query_text": str(query["question"]),
                    "retrieved_titles": merged_ids,
                    "retrieved_texts": [
                        title_to_text.get(cid, "") for cid in merged_ids],
                    # Rank-only score: the two rounds' scores are not on a
                    # shared scale, exactly like the PCST output contract.
                    "retrieved_scores": [
                        1.0 / (rank + 1) for rank in range(len(merged_ids))],
                    "gold_titles": _gold_titles(query),
                    "gold_texts": _optional_gold_texts(query),
                }, ensure_ascii=False) + "\n")
                trace_fh.write(json.dumps(trace_row, ensure_ascii=False) + "\n")
    def _ranked_passage_row(query, position_ids, score_list):
        docs_list = [hipporag.chunk_embedding_store.get_row(
            hipporag.passage_node_keys[int(idx)])["content"] for idx in position_ids]
        return {
            "query_id": _adapter_query_id(query),
            "query_text": str(query["question"]),
            "retrieved_titles": [text_to_title.get(str(t), "") for t in docs_list],
            "retrieved_texts": docs_list,
            "retrieved_scores": [float(x) for x in score_list],
            "gold_titles": _gold_titles(query),
            "gold_texts": _optional_gold_texts(query),
        }

    def _official_row(query, official_solution):
        docs_list = list(official_solution.docs)
        return {
            "query_id": _adapter_query_id(query),
            "query_text": str(query["question"]),
            "retrieved_titles": [text_to_title.get(str(t), "") for t in docs_list],
            "retrieved_texts": docs_list,
            "retrieved_scores": [float(x) for x in official_solution.doc_scores],
            "gold_titles": _gold_titles(query),
            "gold_texts": _optional_gold_texts(query),
        }

    spreading_outputs: dict[str, str] = {}
    spreading_trace_path = out_path.with_suffix(".spreading_trace.jsonl")
    if spreading_variants:
        spreading_runtime["retrieval_queries"] = len(queries)
        spreading_runtime["dense_fallback_queries"] = sum(
            str(query["question"]) not in spreading_sink for query in queries)
        with spreading_trace_path.open("w", encoding="utf-8") as trace_fh:
            for query in queries:
                query_text = str(query["question"])
                payload = spreading_sink.get(query_text, {})
                trace_fh.write(json.dumps({
                    "query_id": _adapter_query_id(query), "query_text": query_text,
                    "dense_fallback": not bool(payload),
                    "variants": {name: entry["diagnostics"]
                                 for name, entry in payload.items()},
                }, ensure_ascii=False) + "\n")
        for variant in spreading_variants:
            variant_path = out_path.with_name(
                f"{out_path.stem}.spreading_{variant}{out_path.suffix}")
            with variant_path.open("w", encoding="utf-8") as fh:
                for query, official_solution in zip(queries, solutions, strict=True):
                    payload = spreading_sink.get(
                        str(query["question"]), {}).get(variant)
                    row = (_official_row(query, official_solution)
                           if payload is None else _ranked_passage_row(
                               query, payload["doc_ids"], payload["doc_scores"]))
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            spreading_outputs[variant] = str(variant_path)
    multi_pcst_outputs: dict[str, str] = {}
    multi_pcst_trace_path = out_path.with_suffix(".multi_pcst_trace.jsonl")
    if multi_pcst_variants:
        multi_pcst_runtime["retrieval_queries"] = len(queries)
        multi_pcst_runtime["dense_fallback_queries"] = sum(
            str(query["question"]) not in multi_pcst_sink for query in queries)
        with multi_pcst_trace_path.open("w", encoding="utf-8") as trace_fh:
            for query in queries:
                query_text = str(query["question"])
                payload = multi_pcst_sink.get(query_text, {})
                trace_fh.write(json.dumps({
                    "query_id": _adapter_query_id(query), "query_text": query_text,
                    "dense_fallback": not bool(payload),
                    "variants": {name: entry["diagnostics"]
                                 for name, entry in payload.items()},
                }, ensure_ascii=False) + "\n")
        for variant in multi_pcst_variants:
            variant_path = out_path.with_name(
                f"{out_path.stem}.multi_pcst_{variant}{out_path.suffix}")
            with variant_path.open("w", encoding="utf-8") as fh:
                for query, official_solution in zip(queries, solutions, strict=True):
                    payload = multi_pcst_sink.get(
                        str(query["question"]), {}).get(variant)
                    row = (_official_row(query, official_solution)
                           if payload is None else _ranked_passage_row(
                               query, payload["doc_ids"], payload["doc_scores"]))
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            multi_pcst_outputs[variant] = str(variant_path)
    bridge_outputs: dict[str, str] = {}
    bridge_trace_path = out_path.with_suffix(".dense_bridge_trace.jsonl")
    bridge_runtime = None
    bridge_params_obj = None
    if dense_bridge:
        from candidate_pool.retrieval.dense_graph_bridge import (
            VARIANTS as BRIDGE_VARIANTS, BridgeParameters, DenseGraphBridge,
        )
        from candidate_pool.retrieval.spreading_activation import (
            SpreadingActivation, SpreadingParameters,
        )
        bridge_params_obj = BridgeParameters(**dict(dense_bridge_params or {}))
        bridge = DenseGraphBridge(
            bridge_adapter,
            SpreadingActivation(
                bridge_adapter, SpreadingParameters(**dict(spreading_params or {}))),
            bridge_params_obj)
        position_by_text: dict[str, int] = {}
        for position, key in enumerate(hipporag.passage_node_keys):
            content = hipporag.chunk_embedding_store.get_row(key)["content"]
            position_by_text.setdefault(content, position)
        bridge_runtime = {"retrieval_queries": len(queries),
                          "graph_entered_queries": 0,
                          "bridged_queries": 0, "seconds": 0.0}
        bridge_rows: dict[str, list[dict]] = {v: [] for v in BRIDGE_VARIANTS}
        bridge_trace_rows: list[dict] = []
        for query, official_solution in zip(queries, solutions, strict=True):
            query_text = str(query["question"])
            if query_text in bridge_entered:
                bridge_runtime["graph_entered_queries"] += 1
                for variant in BRIDGE_VARIANTS:
                    bridge_rows[variant].append(
                        _official_row(query, official_solution))
                bridge_trace_rows.append({
                    "query_id": _adapter_query_id(query),
                    "query_text": query_text,
                    "graph_entered": True, "bridged": False,
                })
                continue
            dense_positions = [
                position_by_text[str(t)] for t in official_solution.docs
                if str(t) in position_by_text]
            dense_scores = [float(x) for x in official_solution.doc_scores]
            trace_entry = {"query_id": _adapter_query_id(query),
                           "query_text": query_text,
                           "graph_entered": False, "bridged": True,
                           "variants": {}}
            for variant in BRIDGE_VARIANTS:
                positions, ranked_scores, diagnostics = bridge.retrieve(
                    query_text, dense_positions, dense_scores, top_k, variant)
                bridge_runtime["seconds"] += float(diagnostics["seconds"])
                bridge_rows[variant].append(
                    _ranked_passage_row(query, positions, ranked_scores))
                trace_entry["variants"][variant] = diagnostics
            bridge_runtime["bridged_queries"] += 1
            bridge_trace_rows.append(trace_entry)
        with bridge_trace_path.open("w", encoding="utf-8") as trace_fh:
            for row in bridge_trace_rows:
                trace_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        for variant, rows_list in bridge_rows.items():
            variant_path = out_path.with_name(
                f"{out_path.stem}.{variant}{out_path.suffix}")
            with variant_path.open("w", encoding="utf-8") as fh:
                for row in rows_list:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            bridge_outputs[variant] = str(variant_path)
    trace_path = out_path.with_suffix(".recognition_trace.jsonl")
    if recognition_trace:
        query_ids_by_text = {
            str(query["question"]): _adapter_query_id(query) for query in queries
        }
        with trace_path.open("w", encoding="utf-8") as fh:
            for trace in recognition_trace:
                fh.write(json.dumps({
                    "query_id": query_ids_by_text.get(trace["query_text"], ""),
                    **trace,
                }, ensure_ascii=False) + "\n")
    entry_trace_path = out_path.with_suffix(".entry_trace.jsonl")
    if entry_trace:
        with entry_trace_path.open("w", encoding="utf-8") as fh:
            for query in queries:
                trace = entry_trace_by_query.get(str(query["question"]), {})
                fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
    ontology_trace_path = out_path.with_suffix(".ontology_trace.jsonl")
    if ontology_trace_by_query:
        with ontology_trace_path.open("w", encoding="utf-8") as fh:
            for query in queries:
                query_text = str(query["question"])
                fh.write(json.dumps({
                    "query_id": _adapter_query_id(query),
                    "query_text": query_text,
                    **ontology_trace_by_query.get(query_text, {}),
                }, ensure_ascii=False) + "\n")
    manifest = {
        "protocol": "official HippoRAG end-to-end validation baseline",
        "method_version": "official-fixed-graph-query-time-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **_git_provenance(),
        "random_seed": None,
        "test_split_used": bool(test_split_used),
        "upstream": "OSU-NLP-Group/HippoRAG main",
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
        "section17_seed_mixture": seed_mixture_manifest,
        "ontology_augmentation": ontology_manifest,
        "ontology_trace": (
            str(ontology_trace_path) if ontology_trace_by_query else None),
        "ontology_trace_queries": len(ontology_trace_by_query),
        "evidence_requirements": evidence_manifest,
        "only_labelled_queries": only_labelled_queries,
        "recognition_trace": str(trace_path) if recognition_trace else None,
        "recognition_trace_queries": len(recognition_trace),
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
        "environment_note": "Use upstream main + LiteLLM/Boto3; direct class import is required by upstream's current package export.",
    }
    if transition_profiles:
        state_after = transition_adapter.state_signature()
        manifest["official_transition_profiles"] = {
            "enabled": True,
            "profiles": transition_profiles,
            "parameters": {
                "alpha": transition_alpha, "beta": transition_beta,
                "hub_gamma": transition_hub_gamma, "epsilon": transition_epsilon,
                "relation_aggregation": transition_relation_aggregation,
            },
            "outputs": transition_outputs,
            "trace": str(transition_trace_path),
            "runtime": transition_runtime,
            "relation_sidecar": transition_sidecar_manifest,
            "relation_embeddings": transition_relation_embedding_manifest,
            "graph_state_before": transition_state_before,
            "graph_state_after": state_after,
            "graph_immutable": transition_state_before == state_after,
            "transition_extra_llm_calls": 0,
            "transition_extra_embedding_calls": (
                0 if transition_relation_embedding_manifest is None
                or transition_relation_embedding_manifest.get("cache_hit") else
                math.ceil(transition_relation_embedding_manifest["labels"] / 64)),
            "recognition_shared_once": True,
        }
    if local_pcst:
        pcst_state_after = pcst_adapter.state_signature()
        manifest["official_local_pcst"] = {
            "enabled": True, "method_name": "Official-Hippo local PCST adaptation",
            "paper_reproduction": False,
            "learned_g_retriever_encoder_used": False,
            "parameters": dict(local_pcst_params or {}),
            "output": str(pcst_output_path), "trace": str(pcst_trace_path),
            "runtime": pcst_runtime, "graph_state_before": pcst_state_before,
            "graph_state_after": pcst_state_after,
            "graph_immutable": pcst_state_before == pcst_state_after,
            "extra_llm_calls": 0, "extra_embedding_calls": 0,
            "recognition_shared_once": True,
        }
    if ircot:
        manifest["official_ircot"] = {
            "enabled": True,
            "method_name": "one-step IRCoT adaptation on Official HippoRAG2",
            "upstream_reference": "StonyBrookNLP/ircot (Trivedi et al., ACL 2023)",
            "paper_reproduction": False,
            "first_retriever": "official HippoRAG2 retrieve (shared instance)",
            "second_retriever": "identical to first (same instance, same top_k)",
            "parameters": {
                "prompt_context_count": ircot_params_obj.prompt_context_count,
                "max_para_num_words": ircot_params_obj.max_para_num_words,
                "answer_extractor_regex": ircot_params_obj.answer_extractor_regex,
                "merge_strategy": ircot_params_obj.merge_strategy,
                "query_transform": ircot_params_obj.query_transform,
                "prompt_header_path": ircot_params_obj.prompt_header_path,
                "llm_max_tokens": ircot_params_obj.llm_max_tokens,
                "llm_temperature": ircot_params_obj.llm_temperature,
            },
            "output": str(ircot_output_path),
            "trace": str(ircot_trace_path),
            "generation_cache": str(ircot_cache.path),
            "runtime": ircot_stats,
            "extra_llm_calls_cot": ircot_stats["llm_calls"],
            "second_round_official_recognition_llm_calls": (
                ircot_stats["second_retrieval_queries"]),
            "graph_pickle_sha256": ircot_graph_sha,
            "graph_immutable": ircot_graph_sha["before"] == ircot_graph_sha["after"],
            "cot_disclosure": (
                "full CoT generation (first 600 chars) and the public second "
                "query are both archived in the trace"),
        }
    if spreading_variants:
        state_after = spreading_adapter.state_signature()
        manifest["official_spreading"] = {
            "enabled": True,
            "method_name": "fixed-hop query-aware spreading activation",
            "variants": list(spreading_variants),
            "parameters": dict(spreading_params or {}),
            "outputs": spreading_outputs,
            "trace": str(spreading_trace_path),
            "runtime": spreading_runtime,
            "graph_state_before": spreading_state_before,
            "graph_state_after": state_after,
            "graph_immutable": spreading_state_before == state_after,
            "extra_llm_calls": 0,
            "recognition_shared_once": True,
        }
    if multi_pcst_variants:
        state_after = multi_pcst_adapter.state_signature()
        manifest["official_multi_pcst"] = {
            "enabled": True,
            "method_name": "diversified multi-tree PCST on Official graph",
            "variants": list(multi_pcst_variants),
            "parameters": dict(multi_pcst_params or {}),
            "local_pcst_parameters": dict(local_pcst_params or {}),
            "outputs": multi_pcst_outputs,
            "trace": str(multi_pcst_trace_path),
            "runtime": multi_pcst_runtime,
            "graph_state_before": multi_pcst_state_before,
            "graph_state_after": state_after,
            "graph_immutable": multi_pcst_state_before == state_after,
            "extra_llm_calls": 0,
            "recognition_shared_once": True,
        }
    if dense_bridge:
        state_after = bridge_adapter.state_signature()
        manifest["official_dense_bridge"] = {
            "enabled": True,
            "method_name": "ToG-style dense-to-graph bridge (fallback queries)",
            "variants": ["dense_bridge_1hop", "dense_bridge_2hop",
                         "dense_bridge_relation"],
            "parameters": dict(dense_bridge_params or {}),
            "spreading_parameters": dict(spreading_params or {}),
            "outputs": bridge_outputs,
            "trace": str(bridge_trace_path),
            "runtime": bridge_runtime,
            "graph_state_before": bridge_state_before,
            "graph_state_after": state_after,
            "graph_immutable": bridge_state_before == state_after,
            "extra_llm_calls": 0,
            "recognition_shared_once": True,
            "pure_dense_control": "official output file (fallback rows are dense)",
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

    recognition_cfg = project_config.params("hipporag_recognition", default={})
    entry_cfg = project_config.params("hipporag_entry", default={})
    ontology_cfg = project_config.params("hipporag_ontology_bridge", default={})
    transition_cfg = project_config.params("official_transition_profiles", default={})
    pcst_cfg = project_config.params("official_local_pcst", default={})
    ircot_cfg = project_config.params("official_ircot", default={})
    bridge_cfg = project_config.params("official_dense_bridge", default={})
    spreading_cfg = project_config.params("official_spreading", default={})
    multi_pcst_cfg = project_config.params("official_multi_pcst", default={})
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
        default="official",
        help="Retrieval-only ablation; all profiles reuse the same indexed graph.",
    )
    ap.add_argument("--evidence-requirements", type=Path,
                    help="Blind labels.csv used only by adhd_need_aware/query filtering.")
    ap.add_argument("--only-labelled-queries", action="store_true",
                    help="Restrict validation evaluation to agreed labelled query ids.")
    ap.add_argument("--default-evidence-type", default=str(
        recognition_cfg.get("default_evidence_type", "direct_match")))
    ap.add_argument(
        "--llm-model",
        default="bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    ap.add_argument("--embedding-model", default="cohere.embed-multilingual-v3")
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
    ap.add_argument(
        "--seed-mixture-decompositions", type=Path,
        help="Section17 query-facet JSONL; mixes fact+passage scores before official PPR.",
    )
    ap.add_argument("--entry-trace-top-k", type=int, default=int(
        entry_cfg.get("trace_top_k", 20)))
    ap.add_argument("--fact-seed-temperature", type=float, default=float(
        entry_cfg.get("fact_seed_temperature", 0.10)))
    ap.add_argument(
        "--ontology-spec", type=Path,
        default=Path(str(ontology_cfg.get(
            "spec", "configuration/adhd_theory_ontology.json"))),
        help="Literature ontology spine used only by ontology_* profiles.",
    )
    ap.add_argument("--ontology-top-k-per-concept", type=int, default=int(
        ontology_cfg.get("top_k_per_concept", 12)))
    ap.add_argument("--ontology-min-similarity", type=float, default=float(
        ontology_cfg.get("min_similarity", 0.35)))
    ap.add_argument("--ontology-max-concepts-per-entity", type=int, default=int(
        ontology_cfg.get("max_concepts_per_entity", 2)))
    ap.add_argument("--ontology-bridge-weight", type=float, default=float(
        ontology_cfg.get("bridge_weight", 0.20)))
    ap.add_argument("--ontology-hierarchy-weight", type=float, default=float(
        ontology_cfg.get("hierarchy_weight", 0.05)))
    ap.add_argument("--ontology-seed-weight", type=float, default=float(
        ontology_cfg.get("seed_weight", 0.10)))
    ap.add_argument("--ontology-seed-top-k", type=int, default=int(
        ontology_cfg.get("seed_top_k", 3)))
    ap.add_argument(
        "--transition-profiles",
        nargs="+",
        choices=("static", "hub", "node", "node_hub", "node_relation", "node_relation_hub"),
        help=("Rerun only Official PPR transition weights. Recognition, restart "
              "mass and dense fallback are shared with the official run."),
    )
    ap.add_argument(
        "--transition-cache-dir", type=Path,
        help="External cache for relation provenance/embeddings; never the graph directory.",
    )
    ap.add_argument("--transition-alpha", type=float, default=float(
        transition_cfg.get("alpha", 1.0)))
    ap.add_argument("--transition-beta", type=float, default=float(
        transition_cfg.get("beta", 1.0)))
    ap.add_argument("--transition-hub-gamma", type=float, default=float(
        transition_cfg.get("hub_gamma", 0.5)))
    ap.add_argument("--transition-epsilon", type=float, default=float(
        transition_cfg.get("epsilon", 0.05)))
    ap.add_argument(
        "--transition-relation-aggregation", choices=("max", "mean"),
        default=str(transition_cfg.get("relation_aggregation", "max")),
    )
    ap.add_argument(
        "--local-pcst", action="store_true",
        help="Run training-free bounded PCST on the same Official restart vector and graph.",
    )
    ap.add_argument(
        "--ircot", action="store_true",
        help="Run one-step IRCoT (CoT sentence -> second official retrieval -> merge).",
    )
    ap.add_argument("--ircot-cache-dir", type=Path, default=None,
                    help="CoT generation cache dir (default: <out dir>/ircot_cache).")
    ap.add_argument(
        "--dense-bridge", action="store_true",
        help="Run the ToG-style dense-to-graph bridge on recognition-fallback queries "
             "(all three variants).",
    )
    ap.add_argument(
        "--spreading-variants", nargs="+", default=None,
        choices=["spread_node_2hop", "spread_node_hub_2hop",
                 "spread_node_relation_2hop", "spread_node_hub_3hop"],
        help="Run fixed-hop query-aware spreading activation variants.",
    )
    ap.add_argument(
        "--multi-pcst-variants", nargs="+", default=None,
        choices=["pcst_multi2_node_penalty", "pcst_multi2_edge_penalty",
                 "pcst_multi2_passage_penalty", "pcst_component_coverage"],
        help="Run diversified multi-tree PCST variants.",
    )
    ap.add_argument("--stage-cache-dir", type=Path, default=None,
                    help="Shared relation sidecar/embedding cache for round-2 stages.")
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
        evidence_requirements=args.evidence_requirements,
        only_labelled_queries=args.only_labelled_queries,
        soft_rejected_floor=float(recognition_cfg.get(
            "soft_rejected_floor", 0.25)),
        need_aware_floors=dict(recognition_cfg.get(
            "need_aware_rejected_floors", {})),
        default_evidence_type=args.default_evidence_type,
        fact_seed_temperature=args.fact_seed_temperature,
        entry_trace=args.entry_trace,
        entry_trace_top_k=args.entry_trace_top_k,
        ontology_spec=args.ontology_spec,
        ontology_top_k_per_concept=args.ontology_top_k_per_concept,
        ontology_min_similarity=args.ontology_min_similarity,
        ontology_max_concepts_per_entity=args.ontology_max_concepts_per_entity,
        ontology_bridge_weight=args.ontology_bridge_weight,
        ontology_hierarchy_weight=args.ontology_hierarchy_weight,
        ontology_seed_weight=args.ontology_seed_weight,
        ontology_seed_top_k=args.ontology_seed_top_k,
        prepend_title=args.prepend_title,
        openie_cache_provenance=args.openie_cache_provenance,
        seed_mixture_decompositions=args.seed_mixture_decompositions,
        seed_mixture_weights=dict(project_config.params(
            "section17_retrieval_pipeline", "multiquery", "weights", default={})),
        transition_profiles=args.transition_profiles,
        transition_cache_dir=args.transition_cache_dir,
        transition_alpha=args.transition_alpha,
        transition_beta=args.transition_beta,
        transition_hub_gamma=args.transition_hub_gamma,
        transition_epsilon=args.transition_epsilon,
        transition_relation_aggregation=args.transition_relation_aggregation,
        local_pcst=args.local_pcst,
        local_pcst_params=dict(pcst_cfg),
        ircot=args.ircot,
        ircot_params=dict(ircot_cfg),
        ircot_cache_dir=args.ircot_cache_dir,
        dense_bridge=args.dense_bridge,
        dense_bridge_params=dict(bridge_cfg),
        spreading_variants=args.spreading_variants,
        spreading_params=dict(spreading_cfg),
        multi_pcst_variants=args.multi_pcst_variants,
        multi_pcst_params=dict(multi_pcst_cfg),
        stage_cache_dir=args.stage_cache_dir,
        test_split_used=args.test_split_used,
        expected_query_sha256=args.expected_query_sha256,
        expected_query_count=args.expected_query_count,
        forbid_llm_inference=args.forbid_llm_inference,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
