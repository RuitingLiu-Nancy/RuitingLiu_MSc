"""In-memory ontology spine for an already indexed HippoRAG 2 graph.

This is an adapted diagnostic inspired by semantic/ontology GraphRAG.  It does
not claim to reproduce Graphwise's proprietary agents or SPARQL pipeline.  The
official OpenIE graph, fact embeddings and passage links remain untouched on
disk; a small literature-anchored ontology is connected to phrase nodes in
memory so the change can be switched off and ablated cleanly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import numpy as np


@dataclass(frozen=True)
class BridgeCandidate:
    concept_id: str
    entity_key: str
    entity_text: str
    similarity: float


def load_ontology_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    concepts = spec.get("concepts") or []
    ids = [str(row.get("id") or "") for row in concepts]
    if not concepts or any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("ontology concepts must have unique non-empty ids")
    known = set(ids)
    for edge in spec.get("hierarchy_edges") or []:
        if edge.get("source") not in known or edge.get("target") not in known:
            raise ValueError(f"ontology hierarchy edge has unknown endpoint: {edge}")
    return spec


def _row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("embedding matrix must be two-dimensional")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms > 0, norms, 1.0)


def select_bridge_candidates(
    concept_ids: list[str],
    concept_embeddings: np.ndarray,
    entity_keys: list[str],
    entity_texts: list[str],
    entity_embeddings: np.ndarray,
    *,
    top_k_per_concept: int = 12,
    min_similarity: float = 0.35,
    max_concepts_per_entity: int = 2,
) -> list[BridgeCandidate]:
    """Return degree-capped semantic concept↔phrase bridges without labels."""
    if top_k_per_concept < 1 or max_concepts_per_entity < 1:
        raise ValueError("bridge degree caps must be positive")
    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be in [-1, 1]")
    if len(concept_ids) != len(concept_embeddings):
        raise ValueError("concept ids/embeddings length mismatch")
    if not (len(entity_keys) == len(entity_texts) == len(entity_embeddings)):
        raise ValueError("entity keys/texts/embeddings length mismatch")
    if not concept_ids or not entity_keys:
        return []

    similarities = _row_normalize(concept_embeddings) @ _row_normalize(entity_embeddings).T
    proposed: list[BridgeCandidate] = []
    for cidx, concept_id in enumerate(concept_ids):
        ranked = np.argsort(similarities[cidx])[::-1][:top_k_per_concept]
        for eidx in ranked:
            score = float(similarities[cidx, eidx])
            if score < min_similarity:
                continue
            proposed.append(BridgeCandidate(
                concept_id=concept_id,
                entity_key=entity_keys[int(eidx)],
                entity_text=entity_texts[int(eidx)],
                similarity=score,
            ))

    # Generic phrases can otherwise connect many theoretical axes and become
    # artificial hubs. Keep only their strongest independently computed links.
    proposed.sort(key=lambda row: (-row.similarity, row.concept_id, row.entity_key))
    entity_degree: dict[str, int] = {}
    accepted: list[BridgeCandidate] = []
    for row in proposed:
        if entity_degree.get(row.entity_key, 0) >= max_concepts_per_entity:
            continue
        accepted.append(row)
        entity_degree[row.entity_key] = entity_degree.get(row.entity_key, 0) + 1
    return sorted(accepted, key=lambda row: (row.concept_id, -row.similarity, row.entity_key))


def mix_theory_reset(
    reset_prob: np.ndarray,
    concept_vertex_indices: list[int],
    concept_scores: np.ndarray,
    theory_seed_weight: float,
    *,
    top_k: int = 3,
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """Mix query→concept reset mass while preserving λ=0 exactly."""
    if not 0.0 <= theory_seed_weight < 1.0:
        raise ValueError("theory_seed_weight must be in [0, 1)")
    output = np.asarray(reset_prob, dtype=float).copy()
    if theory_seed_weight == 0.0 or not concept_vertex_indices:
        return output, []
    scores = np.maximum(np.asarray(concept_scores, dtype=float), 0.0)
    ranked = np.argsort(scores)[::-1][:max(1, top_k)]
    ranked = [int(idx) for idx in ranked if scores[int(idx)] > 0]
    if not ranked:
        return output, []
    base_mass = float(np.maximum(output, 0.0).sum())
    if base_mass <= 0:
        return output, []
    selected = scores[ranked]
    selected = selected / selected.sum()
    added_mass = base_mass * theory_seed_weight / (1.0 - theory_seed_weight)
    trace = []
    for concept_idx, share in zip(ranked, selected, strict=True):
        vertex_idx = concept_vertex_indices[concept_idx]
        mass = float(added_mass * share)
        output[vertex_idx] += mass
        trace.append((concept_idx, mass))
    return output, trace


def augment_hipporag_with_ontology(
    hipporag,
    spec_path: Path,
    *,
    top_k_per_concept: int = 12,
    min_similarity: float = 0.35,
    max_concepts_per_entity: int = 2,
    bridge_weight: float = 0.20,
    hierarchy_weight: float = 0.05,
    theory_seed_weight: float = 0.0,
    theory_seed_top_k: int = 3,
    trace_sink: dict[str, dict] | None = None,
) -> dict:
    """Add a non-persistent ontology spine and optional query concept seeds."""
    if bridge_weight <= 0 or hierarchy_weight < 0:
        raise ValueError("bridge_weight must be positive and hierarchy_weight non-negative")
    spec = load_ontology_spec(spec_path)
    concepts = spec["concepts"]
    concept_ids = [str(row["id"]) for row in concepts]
    concept_texts = [f"{row['label']}: {row['description']}" for row in concepts]

    # No query instruction => Cohere search_document, matching the stored entity
    # embedding side. This is one small batch and does not rerun OpenIE.
    concept_embeddings = np.asarray(
        hipporag.embedding_model.batch_encode(concept_texts, norm=True), dtype=float)
    entity_keys = list(hipporag.entity_node_keys)
    entity_embeddings = np.asarray(hipporag.entity_embeddings, dtype=float)
    vertex_by_name = dict(hipporag.node_name_to_vertex_idx)
    entity_texts = [str(hipporag.graph.vs[vertex_by_name[key]]["content"]) for key in entity_keys]
    bridges = select_bridge_candidates(
        concept_ids, concept_embeddings, entity_keys, entity_texts, entity_embeddings,
        top_k_per_concept=top_k_per_concept,
        min_similarity=min_similarity,
        max_concepts_per_entity=max_concepts_per_entity,
    )

    concept_names = [f"ontology-{concept_id}" for concept_id in concept_ids]
    hipporag.graph.add_vertices(len(concepts), attributes={
        "name": concept_names,
        "hash_id": concept_names,
        "content": concept_texts,
    })
    hipporag.node_name_to_vertex_idx = {
        node["name"]: idx for idx, node in enumerate(hipporag.graph.vs)
    }
    concept_vertex_indices = [hipporag.node_name_to_vertex_idx[name] for name in concept_names]
    concept_vertex_by_id = dict(zip(concept_ids, concept_vertex_indices, strict=True))

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for row in bridges:
        edges.append((concept_vertex_by_id[row.concept_id],
                      hipporag.node_name_to_vertex_idx[row.entity_key]))
        weights.append(float(bridge_weight * row.similarity))
    for row in spec.get("hierarchy_edges") or []:
        edges.append((concept_vertex_by_id[row["source"]], concept_vertex_by_id[row["target"]]))
        weights.append(float(hierarchy_weight))
    if edges:
        hipporag.graph.add_edges(edges, attributes={"weight": weights})

    original_graph_search = hipporag.graph_search_with_fact_entities
    original_run_ppr = hipporag.run_ppr
    active_query: dict[str, str] = {}
    trace_sink = trace_sink if trace_sink is not None else {}

    def _graph_search(self, query, *args, **kwargs):
        active_query["text"] = query
        try:
            return original_graph_search(query, *args, **kwargs)
        finally:
            active_query.pop("text", None)

    def _run_ppr(self, reset_prob, damping=0.5):
        query = active_query.get("text")
        if query and theory_seed_weight > 0:
            query_vector = np.asarray(self.query_to_embedding["triple"][query], dtype=float).reshape(1, -1)
            scores = (_row_normalize(concept_embeddings) @ _row_normalize(query_vector).T).reshape(-1)
            reset_prob, selected = mix_theory_reset(
                reset_prob, concept_vertex_indices, scores, theory_seed_weight,
                top_k=theory_seed_top_k)
            trace_sink[query] = {
                "theory_seed_weight": theory_seed_weight,
                "selected_concepts": [{
                    "concept_id": concept_ids[idx],
                    "label": concepts[idx]["label"],
                    "cosine": float(scores[idx]),
                    "added_reset_mass": mass,
                } for idx, mass in selected],
            }
        return original_run_ppr(reset_prob, damping=damping)

    hipporag.graph_search_with_fact_entities = MethodType(_graph_search, hipporag)
    hipporag.run_ppr = MethodType(_run_ppr, hipporag)
    return {
        "enabled": True,
        "implementation": "in_memory_openie_phrase_ontology_spine_v1",
        "spec": str(spec_path),
        "spec_version": spec.get("version"),
        "concepts": len(concepts),
        "semantic_bridge_edges": len(bridges),
        "hierarchy_edges": len(spec.get("hierarchy_edges") or []),
        "top_k_per_concept": top_k_per_concept,
        "min_similarity": min_similarity,
        "max_concepts_per_entity": max_concepts_per_entity,
        "bridge_weight": bridge_weight,
        "hierarchy_weight": hierarchy_weight,
        "theory_seed_weight": theory_seed_weight,
        "theory_seed_top_k": theory_seed_top_k,
        "persistent_graph_modified": False,
        "label_or_gold_used": False,
        "bridges": [row.__dict__ for row in bridges],
    }
