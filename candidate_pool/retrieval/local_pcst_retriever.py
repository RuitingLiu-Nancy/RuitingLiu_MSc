"""Training-free local PCST retrieval on the immutable Official HippoRAG2 graph.

This is an adaptation of G-Retriever's PCST selection core, not a reproduction
of its learned textual-graph encoder.  It consumes the exact Official restart
vector and node embeddings, solves PCST only on a bounded query-local subgraph,
and never mutates or persists the graph.

The query-local problem construction (anchors -> bounded expansion -> prizes/
costs) and the phrase->passage mapping are exposed as reusable building blocks
(``build_local_problem`` / ``map_selected_to_passages``) so that diversified
multi-tree retrieval (``multi_pcst.py``) reuses this exact code path instead of
maintaining a parallel implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from candidate_pool.retrieval.official_graph_adapter import OfficialGraphAdapter


@dataclass(frozen=True)
class LocalPCSTParameters:
    entity_anchors: int = 6
    passage_anchors: int = 6
    hops: int = 2
    max_local_nodes: int = 2500
    max_neighbors_per_node: int = 100
    query_prize_top_k: int = 32
    passage_prize_top_k: int = 16
    max_mapping_entity_degree: int = 500
    max_passage_candidates: int = 500
    query_prize: float = 1.0
    seed_prize: float = 2.0
    passage_prize: float = 0.5
    anchor_bonus: float = 2.0
    base_edge_cost: float = 0.35
    hub_cost: float = 0.75


@dataclass
class LocalProblem:
    """Query-local PCST instance over official node/edge ids (read-only)."""
    fallback: str | None = None
    anchors: list[int] = field(default_factory=list)
    entity_anchors: list[int] = field(default_factory=list)
    passage_anchors: list[int] = field(default_factory=list)
    local_nodes: list[int] = field(default_factory=list)
    local_index: dict[int, int] = field(default_factory=dict)
    edge_ids: list[int] = field(default_factory=list)
    edge_array: np.ndarray | None = None      # local endpoint index pairs
    prizes: np.ndarray | None = None
    costs: np.ndarray | None = None
    query_scores: np.ndarray | None = None
    reset: np.ndarray | None = None


def solve_pcst(edges: np.ndarray, prizes: np.ndarray, costs: np.ndarray):
    """Thin testable wrapper over the public ``pcst_fast`` API."""
    from pcst_fast import pcst_fast
    return pcst_fast(
        np.asarray(edges, dtype=np.int64), np.asarray(prizes, dtype=np.float64),
        np.asarray(costs, dtype=np.float64), -1, 1, "gw", 0)


class LocalPCSTRetriever:
    """Bounded PCST selector over Official node ids and edge weights."""

    def __init__(self, adapter: OfficialGraphAdapter, params: LocalPCSTParameters):
        self.adapter = adapter
        self.params = params
        self.graph = adapter.graph
        self.entity_nodes = np.asarray(adapter.hipporag.entity_node_idxs, dtype=np.int64)
        self.passage_nodes = np.asarray(adapter.hipporag.passage_node_idxs, dtype=np.int64)
        self.entity_set = set(self.entity_nodes.tolist())
        self.passage_pos = {int(node): i for i, node in enumerate(self.passage_nodes)}

    @staticmethod
    def _top_positive(values: np.ndarray, node_ids: np.ndarray, n: int) -> list[int]:
        subset = np.asarray(values)[node_ids]
        order = np.argsort(subset)[::-1]
        return [int(node_ids[i]) for i in order[:n] if subset[i] > 0]

    def _local_nodes(self, anchors: list[int], query_scores: np.ndarray) -> list[int]:
        selected = set(anchors)
        frontier = set(anchors)
        for _ in range(self.params.hops):
            candidates = set()
            for node in frontier:
                neighbors = self.graph.neighbors(node)
                if len(neighbors) > self.params.max_neighbors_per_node:
                    neighbors = sorted(
                        neighbors, key=lambda idx: query_scores[idx], reverse=True
                    )[:self.params.max_neighbors_per_node]
                candidates.update(neighbors)
            candidates.difference_update(selected)
            remaining = self.params.max_local_nodes - len(selected)
            if remaining <= 0 or not candidates:
                break
            kept = sorted(candidates, key=lambda idx: query_scores[idx], reverse=True)[:remaining]
            selected.update(kept); frontier = set(kept)
        return sorted(selected)

    def build_local_problem(self, query: str, reset_prob: np.ndarray) -> LocalProblem:
        """Anchors, bounded local subgraph, sparse prizes and edge costs."""
        reset = np.where(np.isfinite(reset_prob) & (reset_prob > 0), reset_prob, 0.0)
        query_scores = self.adapter.query_node_similarity(query)
        entity_anchors = self._top_positive(
            reset, self.entity_nodes, self.params.entity_anchors)
        passage_anchors = self._top_positive(
            reset, self.passage_nodes, self.params.passage_anchors)
        anchors = list(dict.fromkeys(entity_anchors + passage_anchors))
        problem = LocalProblem(
            anchors=anchors, entity_anchors=entity_anchors,
            passage_anchors=passage_anchors, query_scores=query_scores,
            reset=reset)
        if not entity_anchors:
            problem.fallback = "no_entity_fact_anchor"
            return problem
        local_nodes = self._local_nodes(anchors, query_scores)
        local_set = set(local_nodes)
        local_index = {node: i for i, node in enumerate(local_nodes)}
        incident_edges = set()
        for node in local_nodes:
            incident_edges.update(self.graph.incident(node))
        edge_ids, endpoints = [], []
        for edge_id in incident_edges:
            src = int(self.adapter.edge_sources[edge_id])
            dst = int(self.adapter.edge_targets[edge_id])
            if src in local_set and dst in local_set:
                edge_ids.append(edge_id)
                endpoints.append((local_index[src], local_index[dst]))
        problem.local_nodes = local_nodes
        problem.local_index = local_index
        problem.edge_ids = edge_ids
        if not endpoints:
            problem.fallback = "empty_local_edges"
            return problem

        local = np.asarray(local_nodes, dtype=np.int64)
        reset_scale = float(reset[np.asarray(anchors)].max()) or 1.0
        prizes = np.zeros(len(local_nodes), dtype=np.float64)
        query_order = np.argsort(query_scores[local])[::-1][
            :self.params.query_prize_top_k]
        prizes[query_order] += self.params.query_prize * query_scores[local][query_order]
        is_passage = np.asarray([node in self.passage_pos for node in local_nodes])
        passage_local = np.flatnonzero(is_passage)
        if len(passage_local):
            passage_order = passage_local[np.argsort(
                query_scores[local][passage_local])[::-1][
                    :self.params.passage_prize_top_k]]
            prizes[passage_order] += (
                self.params.passage_prize * query_scores[local][passage_order])
        for rank, node in enumerate(anchors):
            prizes[local_index[node]] += self.params.seed_prize * reset[node] / reset_scale
            prizes[local_index[node]] += self.params.anchor_bonus * (
                1.0 - rank / (len(anchors) + 1.0))

        edge_array = np.asarray(endpoints, dtype=np.int64)
        original_endpoints = np.asarray(
            [(local_nodes[u], local_nodes[v]) for u, v in endpoints], dtype=np.int64)
        degree_product = ((self.adapter.degrees[original_endpoints[:, 0]] + 1.0) *
                          (self.adapter.degrees[original_endpoints[:, 1]] + 1.0))
        hub = np.log1p(degree_product)
        hub /= float(hub.max()) or 1.0
        original_weights = self.adapter.original_edge_weights[np.asarray(edge_ids)]
        strength_discount = 1.0 / (1.0 + np.log1p(original_weights))
        costs = self.params.base_edge_cost * strength_discount + self.params.hub_cost * hub
        problem.edge_array = edge_array
        problem.prizes = prizes
        problem.costs = costs
        return problem

    def map_selected_to_passages(self, selected_nodes: set[int],
                                 problem: LocalProblem) -> tuple[list[int], dict]:
        """Passage candidates: selected passages + passages of selected phrases.

        Reuses Official topology and passage mapping; generic entities above
        ``max_mapping_entity_degree`` never fan out to their comments.
        """
        query_scores, reset = problem.query_scores, problem.reset
        candidate_nodes = {node for node in selected_nodes if node in self.passage_pos}
        selected_entities = selected_nodes & self.entity_set
        mapping_entities = {
            node for node in selected_entities
            if self.adapter.degrees[node] <= self.params.max_mapping_entity_degree}
        for node in mapping_entities:
            candidate_nodes.update(
                neighbor for neighbor in self.graph.neighbors(node)
                if neighbor in self.passage_pos)
        candidate_positions = sorted(
            (self.passage_pos[node] for node in candidate_nodes),
            key=lambda pos: (
                query_scores[self.passage_nodes[pos]], reset[self.passage_nodes[pos]]),
            reverse=True)[:self.params.max_passage_candidates]
        return candidate_positions, {
            "selected_entities": len(selected_entities),
            "mapping_entities": len(mapping_entities),
            "hub_entities_excluded_from_passage_mapping": (
                len(selected_entities) - len(mapping_entities)),
        }

    def retrieve(self, query: str, reset_prob: np.ndarray,
                 official_ids: np.ndarray, official_scores: np.ndarray,
                 top_k: int) -> tuple[np.ndarray, np.ndarray, dict]:
        started = perf_counter()
        problem = self.build_local_problem(query, reset_prob)
        if problem.fallback == "no_entity_fact_anchor":
            return np.asarray(official_ids[:top_k]), np.asarray(official_scores[:top_k]), {
                "fallback": "no_entity_fact_anchor", "seconds": perf_counter() - started,
                "anchors": len(problem.anchors), "selected_passages": 0,
            }
        if problem.fallback == "empty_local_edges":
            return np.asarray(official_ids[:top_k]), np.asarray(official_scores[:top_k]), {
                "fallback": "empty_local_edges", "seconds": perf_counter() - started,
                "anchors": len(problem.anchors), "local_nodes": len(problem.local_nodes),
                "selected_passages": 0,
            }
        selected_local, selected_edge_local = solve_pcst(
            problem.edge_array, problem.prizes, problem.costs)
        selected_nodes = {problem.local_nodes[int(idx)] for idx in selected_local}
        candidate_positions, mapping_diag = self.map_selected_to_passages(
            selected_nodes, problem)
        output = list(dict.fromkeys(
            candidate_positions + np.asarray(official_ids).tolist()))[:top_k]
        # Rank-only score is explicit: PCST selection is not calibrated to PPR.
        scores = np.asarray([1.0 / (rank + 1) for rank in range(len(output))], dtype=float)
        return np.asarray(output, dtype=np.int64), scores, {
            "fallback": None, "seconds": perf_counter() - started,
            "entity_anchors": problem.entity_anchors,
            "passage_anchors": problem.passage_anchors,
            "local_nodes": len(problem.local_nodes),
            "local_edges": len(problem.edge_ids),
            "pcst_nodes": len(selected_nodes), "pcst_edges": len(selected_edge_local),
            **mapping_diag,
            "selected_passages": len(candidate_positions),
            "official_fill": max(0, len(output) - len(candidate_positions)),
            "prize_min": float(problem.prizes.min()),
            "prize_mean": float(problem.prizes.mean()),
            "prize_max": float(problem.prizes.max()),
            "cost_min": float(problem.costs.min()),
            "cost_mean": float(problem.costs.mean()),
            "cost_max": float(problem.costs.max()),
        }


def install_local_pcst(hipporag, retriever: LocalPCSTRetriever,
                       sink: dict[str, dict], *, top_k: int) -> dict:
    """Capture the exact upstream query/restart and return official results."""
    from types import MethodType
    original_search = hipporag.graph_search_with_fact_entities
    original_run_ppr = hipporag.run_ppr
    runtime = {"graph_search_queries": 0, "pcst_queries": 0, "seconds": 0.0,
               "current_query": None}

    def _search(self, query, *args, **kwargs):
        runtime["graph_search_queries"] += 1; runtime["current_query"] = query
        try:
            return original_search(query, *args, **kwargs)
        finally:
            runtime["current_query"] = None

    def _run_ppr(self, reset_prob, damping=0.5):
        official_ids, official_scores = original_run_ppr(reset_prob, damping)
        query = runtime["current_query"]
        if query is None:
            return official_ids, official_scores
        ids, scores, diagnostics = retriever.retrieve(
            query, reset_prob, official_ids, official_scores, top_k)
        sink[query] = {"doc_ids": ids, "doc_scores": scores,
                       "diagnostics": diagnostics}
        runtime["pcst_queries"] += 1
        runtime["seconds"] += float(diagnostics["seconds"])
        return official_ids, official_scores

    hipporag.graph_search_with_fact_entities = MethodType(_search, hipporag)
    hipporag.run_ppr = MethodType(_run_ppr, hipporag)
    return runtime
