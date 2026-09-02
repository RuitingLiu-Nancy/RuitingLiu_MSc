"""Read-only adapter over the persisted Official HippoRAG2 igraph.

It exposes the exact official node ids, passage mapping and original edge
weights while allowing query-specific *temporary* weight arrays for PPR.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from candidate_pool.retrieval.hub_correction import high_degree_mask, symmetric_degree_penalty
from candidate_pool.retrieval.relation_sidecar import RelationSidecar


PROFILES = {"static", "hub", "node", "node_hub", "node_relation", "node_relation_hub"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OfficialGraphAdapter:
    """Array view of an initialized HippoRAG object; never writes the graph."""

    def __init__(self, hipporag, graph_pickle: Path,
                 relation_sidecar: RelationSidecar | None = None,
                 relation_embeddings: np.ndarray | None = None):
        if not hipporag.ready_to_retrieve:
            raise RuntimeError("call prepare_retrieval_objects before creating adapter")
        self.hipporag = hipporag
        self.graph = hipporag.graph
        self.graph_pickle = Path(graph_pickle)
        self.graph_fingerprint = file_sha256(self.graph_pickle)
        endpoints = np.asarray(self.graph.get_edgelist(), dtype=np.int64)
        self.edge_sources = endpoints[:, 0]
        self.edge_targets = endpoints[:, 1]
        self.original_edge_weights = np.asarray(self.graph.es["weight"], dtype=np.float64)
        self.degrees = np.asarray(self.graph.degree(), dtype=np.float64)
        self.high_degree = high_degree_mask(self.degrees)
        self.relation_sidecar = relation_sidecar
        self.relation_embeddings = (
            np.asarray(relation_embeddings, dtype=np.float32)
            if relation_embeddings is not None else None)
        self._hub_cache: dict[float, np.ndarray] = {}

    def state_signature(self) -> dict:
        weights = np.asarray(self.graph.es["weight"], dtype=np.float64)
        return {
            "vertices": self.graph.vcount(), "edges": self.graph.ecount(),
            "weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
            "graph_pickle_sha256": file_sha256(self.graph_pickle),
        }

    def query_node_similarity(self, query: str) -> np.ndarray:
        triple = np.asarray(self.hipporag.query_to_embedding["triple"][query]).reshape(-1)
        passage = np.asarray(self.hipporag.query_to_embedding["passage"][query]).reshape(-1)
        scores = np.zeros(self.graph.vcount(), dtype=np.float64)
        entity_scores = np.asarray(self.hipporag.entity_embeddings @ triple).reshape(-1)
        passage_scores = np.asarray(self.hipporag.passage_embeddings @ passage).reshape(-1)
        scores[np.asarray(self.hipporag.entity_node_idxs, dtype=np.int64)] = entity_scores
        scores[np.asarray(self.hipporag.passage_node_idxs, dtype=np.int64)] = passage_scores
        return np.clip(scores, 0.0, 1.0)

    def query_relation_similarity(self, query: str) -> np.ndarray:
        if self.relation_sidecar is None or self.relation_embeddings is None:
            raise RuntimeError("node_relation profile requires sidecar and relation embeddings")
        vector = np.asarray(self.hipporag.query_to_embedding["triple"][query]).reshape(-1)
        return np.clip(np.asarray(self.relation_embeddings @ vector).reshape(-1), 0.0, 1.0)

    def transition_weights(self, query: str, profile: str, *, alpha: float = 1.0,
                           beta: float = 1.0, hub_gamma: float = 0.5,
                           epsilon: float = 0.05,
                           relation_aggregation: str = "max") -> tuple[np.ndarray, dict]:
        if profile not in PROFILES:
            raise ValueError(f"unknown transition profile: {profile}")
        if min(alpha, beta, hub_gamma) < 0 or epsilon <= 0:
            raise ValueError("alpha/beta/hub_gamma must be non-negative and epsilon positive")
        start = perf_counter()
        weights = self.original_edge_weights.copy()
        uses_hub = profile in {"hub", "node_hub", "node_relation_hub"}
        uses_node = profile in {"node", "node_hub", "node_relation", "node_relation_hub"}
        uses_relation = profile in {"node_relation", "node_relation_hub"}
        if uses_hub:
            if hub_gamma not in self._hub_cache:
                self._hub_cache[hub_gamma] = symmetric_degree_penalty(
                    self.degrees, self.edge_sources, self.edge_targets, hub_gamma)
            weights *= self._hub_cache[hub_gamma]
        node_scores = None
        if uses_node:
            node_scores = self.query_node_similarity(query)
            factor = np.power(epsilon + node_scores[self.edge_sources], alpha)
            factor *= np.power(epsilon + node_scores[self.edge_targets], alpha)
            weights *= factor
        relation_covered = 0
        if uses_relation:
            relation_scores = self.query_relation_similarity(query)
            per_edge = self.relation_sidecar.aggregate(relation_scores, relation_aggregation)
            mask = np.isfinite(per_edge)
            relation_covered = int(mask.sum())
            weights[mask] *= np.power(epsilon + per_edge[mask], beta)
        if not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
            raise RuntimeError("transition profile produced invalid edge weights")
        return weights, {
            "profile": profile, "alpha": alpha, "beta": beta,
            "hub_gamma": hub_gamma, "epsilon": epsilon,
            "uses_hub": uses_hub, "uses_node": uses_node,
            "uses_relation": uses_relation,
            "relation_aggregation": relation_aggregation,
            "relation_edges_covered": relation_covered,
            "edge_weight_min": float(weights.min()),
            "edge_weight_mean": float(weights.mean()),
            "edge_weight_max": float(weights.max()),
            "edge_weight_nonzero": int(np.count_nonzero(weights)),
            "node_similarity_mean": float(node_scores.mean()) if node_scores is not None else None,
            "seconds": perf_counter() - start,
        }

    def run_ppr(self, reset_prob: np.ndarray, edge_weights: np.ndarray,
                damping: float) -> tuple[np.ndarray, np.ndarray, dict]:
        start = perf_counter()
        reset = np.asarray(reset_prob, dtype=np.float64)
        reset = np.where(np.isnan(reset) | (reset < 0), 0.0, reset)
        pagerank = np.asarray(self.graph.personalized_pagerank(
            vertices=range(self.graph.vcount()), damping=damping,
            # python-igraph 0.11 treats an ndarray as an edge-attribute key;
            # a plain list is the supported temporary weight-vector form.
            directed=False, weights=np.asarray(edge_weights, dtype=np.float64).tolist(),
            reset=reset, implementation="prpack"), dtype=np.float64)
        passage_scores = pagerank[np.asarray(self.hipporag.passage_node_idxs, dtype=np.int64)]
        order = np.argsort(passage_scores)[::-1]
        top_nodes = np.argsort(pagerank)[::-1][:10]
        names, contents = self.graph.vs["name"], self.graph.vs["content"]
        return order, passage_scores[order], {
            "seconds": perf_counter() - start,
            "top10_node_mass": float(pagerank[top_nodes].sum()),
            "high_degree_mass": float(pagerank[self.high_degree].sum()),
            "top_ppr_hubs": [{
                "node_index": int(idx), "name": str(names[idx]),
                "content": str(contents[idx])[:160], "degree": int(self.degrees[idx]),
                "mass": float(pagerank[idx]),
            } for idx in top_nodes],
        }
