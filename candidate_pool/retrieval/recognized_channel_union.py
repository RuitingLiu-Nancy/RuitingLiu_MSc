"""Official-recognition-preserving graph-entry channel retrieval and union.

Every channel independently executes ``get_fact_scores`` and ``rerank_facts``.
Only channels with retained facts may contribute to a rewrite union. Restart
vectors are constructed with the exact Official endpoint specificity and dense
passage teleport rules, normalized per channel, and mixed at fixed weights.
The persisted graph and Official recognition/PPR implementations are untouched.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def _min_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high == low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _entity_key(phrase: str) -> str:
    return "entity-" + hashlib.md5(phrase.encode()).hexdigest()


@dataclass
class ChannelResult:
    text: str
    condition: str
    channel_id: str
    recognized_fact_count: int
    restart: np.ndarray
    doc_ids: np.ndarray
    doc_scores: np.ndarray
    trace: dict

    @property
    def recognition_success(self) -> bool:
        return self.recognized_fact_count > 0


def build_official_restart(hipporag, query: str, fact_scores: np.ndarray,
                           fact_indices: list[int], facts: list[tuple]) -> tuple[np.ndarray, dict]:
    """Reproduce upstream graph-search reset construction without running PPR."""
    node_count = len(hipporag.graph.vs["name"])
    phrase_weights = np.zeros(node_count, dtype=float)
    occurrence = np.zeros(node_count, dtype=float)
    seed_rows = []
    for fact_index, fact in zip(fact_indices, facts, strict=True):
        score = float(fact_scores[fact_index])
        for phrase in (str(fact[0]).lower(), str(fact[2]).lower()):
            key = _entity_key(phrase)
            vertex = hipporag.node_name_to_vertex_idx.get(key)
            degree = len(hipporag.ent_node_to_chunk_ids.get(key, set()))
            adjusted = score / degree if degree else None
            if vertex is not None:
                phrase_weights[vertex] += score / degree if degree else score
                occurrence[vertex] += 1
            seed_rows.append({
                "phrase": phrase, "entity_id": key,
                "vertex_id": int(vertex) if vertex is not None else None,
                "document_frequency": int(degree),
                "fact_score": score, "specificity_adjusted_input": adjusted,
            })
    nonzero = occurrence > 0
    phrase_weights[nonzero] /= occurrence[nonzero]
    if hipporag.global_config.linking_top_k:
        keep = np.argsort(-phrase_weights, kind="stable")[:
            int(hipporag.global_config.linking_top_k)]
        mask = np.zeros(node_count, dtype=bool)
        mask[keep] = phrase_weights[keep] > 0
        phrase_weights[~mask] = 0

    dense_ids, dense_scores = hipporag.dense_passage_retrieval(query)
    normalized_dense = _min_max(np.asarray(dense_scores, dtype=float))
    passage_weights = np.zeros(node_count, dtype=float)
    passage_rows = []
    passage_factor = float(hipporag.global_config.passage_node_weight)
    for rank, (doc_id, score) in enumerate(
            zip(np.asarray(dense_ids, dtype=int), normalized_dense, strict=True), 1):
        passage_key = hipporag.passage_node_keys[int(doc_id)]
        vertex = hipporag.node_name_to_vertex_idx[passage_key]
        passage_weights[vertex] = float(score) * passage_factor
        if rank <= 20:
            passage_rows.append({"dense_rank": rank, "passage_id": passage_key,
                                 "vertex_id": int(vertex),
                                 "normalized_dense_score": float(score),
                                 "restart_input": float(score) * passage_factor})
    restart = phrase_weights + passage_weights
    return restart, {
        "retained_fact_count": len(fact_indices),
        "phrase_seed_count": int(np.count_nonzero(phrase_weights)),
        "passage_seed_count": int(np.count_nonzero(passage_weights)),
        "restart_nonzero": int(np.count_nonzero(restart)),
        "restart_sum": float(restart.sum()),
        "fact_endpoint_seeds": seed_rows,
        "dense_passage_top20": passage_rows,
    }


def run_independent_channel(hipporag, *, text: str, condition: str,
                            channel_id: str, top_k: int) -> ChannelResult:
    """Execute one exact upstream recognition path and its retrieval result."""
    fact_scores = hipporag.get_fact_scores(text)
    indices, facts, recognition_log = hipporag.rerank_facts(text, fact_scores)
    restart, trace = build_official_restart(
        hipporag, text, fact_scores, indices, facts)
    if indices:
        doc_ids, doc_scores = hipporag.run_ppr(
            restart, damping=hipporag.global_config.damping)
        fallback = False
    else:
        doc_ids, doc_scores = hipporag.dense_passage_retrieval(text)
        fallback = True
    trace.update({
        "condition": condition, "channel_id": channel_id, "query_text": text,
        "recognition_success": bool(indices), "fallback_status": fallback,
        "recognition_filter_log": {
            key: value for key, value in recognition_log.items()
            if key not in {"facts_before_rerank", "facts_after_rerank"}
        },
        "recognition_filter_input": [
            list(fact) for fact in recognition_log.get("facts_before_rerank", [])
        ],
        "recognition_filter_output": [
            list(fact) for fact in recognition_log.get("facts_after_rerank", facts)
        ],
        "retained_facts": [list(fact) for fact in facts],
    })
    return ChannelResult(
        text=text, condition=condition, channel_id=channel_id,
        recognized_fact_count=len(indices), restart=restart,
        doc_ids=np.asarray(doc_ids[:top_k], dtype=int),
        doc_scores=np.asarray(doc_scores[:top_k], dtype=float), trace=trace)


def normalized_union(parts: list[tuple[ChannelResult, float]]) -> np.ndarray:
    """Deduplicate same nodes by vector addition after per-channel normalization."""
    if not parts:
        raise ValueError("union requires at least one channel")
    shape = parts[0][0].restart.shape
    mixed = np.zeros(shape, dtype=float)
    total_weight = sum(float(weight) for _, weight in parts)
    if total_weight <= 0:
        raise ValueError("union channel weights must be positive")
    for channel, weight in parts:
        if channel.restart.shape != shape:
            raise ValueError("channel restart shape mismatch")
        mass = float(channel.restart.sum())
        if mass <= 0:
            continue
        mixed += (float(weight) / total_weight) * channel.restart / mass
    if mixed.sum() <= 0:
        raise ValueError("union restart has no positive mass")
    return mixed


def run_original_plus_rewrites(hipporag, original: ChannelResult,
                               rewrites: list[ChannelResult], *, top_k: int,
                               rewrite_family: str) -> dict:
    """Run pre-registered original:rewrite=0.5:0.5 recognized seed union."""
    successful = [row for row in rewrites if row.recognition_success]
    if not successful:
        return {
            "doc_ids": original.doc_ids, "doc_scores": original.doc_scores,
            "fallback_status": not original.recognition_success,
            "graph_entry_status": original.recognition_success,
            "union_applied": False, "reason": "no_successful_rewrite_channel",
            "channels": [original.channel_id],
            "graph_propagation_executed": original.recognition_success,
            "candidate_score_origin": (
                "official_ppr" if original.recognition_success else "dense_fallback"),
        }
    rewrite_weight = 0.5 / len(successful)
    parts = [(original, 0.5), *[(row, rewrite_weight) for row in successful]]
    reset = normalized_union(parts)
    ids, scores = hipporag.run_ppr(reset, damping=hipporag.global_config.damping)
    return {
        "doc_ids": np.asarray(ids[:top_k], dtype=int),
        "doc_scores": np.asarray(scores[:top_k], dtype=float),
        "fallback_status": False, "graph_entry_status": True,
        "union_applied": True, "reason": None,
        "rewrite_family": rewrite_family,
        "channels": [row.channel_id for row, _ in parts],
        "channel_weights": {row.channel_id: weight for row, weight in parts},
        "restart_nonzero": int(np.count_nonzero(reset)),
        "restart_sum": float(reset.sum()),
        "graph_propagation_executed": True,
        "candidate_score_origin": "recognized_channel_union_ppr",
    }
