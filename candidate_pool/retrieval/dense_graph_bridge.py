"""ToG-style dense-to-graph bridge for recognition-gate fallback queries.

Problem: 71/100 frozen-dev queries fail the official HippoRAG2 recognition
gate and never reach the graph.  Hypothesis (Think-on-Graph style): dense
retrieval still finds semantically related passages, and the phrase/entity
nodes attached to those passages are usable graph entry points.

Provenance: the passage->entity anchoring and bounded exploration follow the
entity-anchored graph exploration idea of Think-on-Graph / ToG-2
(IDEA-FinAI/ToG-2); their LLM-driven pruning agent is NOT reproduced (this
project's boundary: no extra LLM calls in retrieval).  Expansion reuses this
repository's canonical bounded propagator (``SpreadingActivation``) so there
is exactly one propagation implementation.  Recorded as ``adapted`` in
docs_v2/11 §F.

Anchor score (all signals read from existing official stores)::

    score(e) = [sum of dense scores of top-D passages adjacent to e]
               * (anchor_epsilon + s(q, e))
               * (deg(e) + 1) ** (-anchor_degree_gamma)

The dense-score sum is a score-weighted frequency: an entity appearing in
several top dense passages accumulates their scores.  Entities with degree
above ``anchor_max_degree`` are excluded as generic hubs (archived per query).

The graph is never written; the official dense ranking is never replaced --
the bridge output is a merge, and the pure dense run remains the control.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from candidate_pool.retrieval.official_graph_adapter import OfficialGraphAdapter
from candidate_pool.retrieval.spreading_activation import SpreadingActivation


VARIANTS = {
    "dense_bridge_1hop": {"hops": 1, "use_relation": False},
    "dense_bridge_2hop": {"hops": 2, "use_relation": False},
    "dense_bridge_relation": {"hops": 2, "use_relation": True},
}


@dataclass(frozen=True)
class BridgeParameters:
    """Read from configuration/params.yaml::official_dense_bridge."""
    dense_anchor_passages: int = 20   # top-D dense passages considered
    max_anchors: int = 8
    anchor_max_degree: int = 500      # generic-hub exclusion, PCST-consistent
    anchor_epsilon: float = 0.05
    anchor_degree_gamma: float = 0.5
    merge_strategy: str = "interleave"  # graph list vs dense list


class DenseGraphBridge:
    def __init__(self, adapter: OfficialGraphAdapter,
                 activation: SpreadingActivation, params: BridgeParameters):
        self.adapter = adapter
        self.activation = activation
        self.params = params
        self.graph = adapter.graph
        self.passage_nodes = np.asarray(
            adapter.hipporag.passage_node_idxs, dtype=np.int64)
        self.entity_set = set(
            int(x) for x in adapter.hipporag.entity_node_idxs)

    def select_anchors(self, query: str, dense_positions: list[int],
                       dense_scores: list[float]) -> tuple[list[dict], dict]:
        top = list(zip(dense_positions, dense_scores))[
            :self.params.dense_anchor_passages]
        dense_mass: dict[int, float] = {}
        frequency: dict[int, int] = {}
        for position, score in top:
            passage_node = int(self.passage_nodes[position])
            for neighbor in self.graph.neighbors(passage_node):
                if neighbor not in self.entity_set:
                    continue
                dense_mass[neighbor] = dense_mass.get(neighbor, 0.0) + float(score)
                frequency[neighbor] = frequency.get(neighbor, 0) + 1
        query_scores = self.adapter.query_node_similarity(query)
        excluded_hubs = []
        scored = []
        for node, mass in dense_mass.items():
            degree = float(self.adapter.degrees[node])
            if degree > self.params.anchor_max_degree:
                excluded_hubs.append({"node": int(node), "degree": int(degree)})
                continue
            score = (mass
                     * (self.params.anchor_epsilon + float(query_scores[node]))
                     * (degree + 1.0) ** (-self.params.anchor_degree_gamma))
            scored.append({
                "node": int(node),
                "anchor_score": float(score),
                "dense_mass": float(mass),
                "frequency": int(frequency[node]),
                "degree": int(degree),
                "query_similarity": float(query_scores[node]),
                "content": str(self.graph.vs[node]["content"])[:80],
            })
        scored.sort(key=lambda row: row["anchor_score"], reverse=True)
        anchors = scored[:self.params.max_anchors]
        return anchors, {
            "candidate_entities": len(dense_mass),
            "excluded_hub_count": len(excluded_hubs),
            "excluded_hub_sample": excluded_hubs[:5],
            "anchor_passages_used": len(top),
        }

    def retrieve(self, query: str, dense_positions: list[int],
                 dense_scores: list[float], top_k: int,
                 variant: str) -> tuple[list[int], list[float], dict]:
        if variant not in VARIANTS:
            raise ValueError(f"unknown dense bridge variant: {variant}")
        config = VARIANTS[variant]
        started = perf_counter()
        anchors, anchor_diag = self.select_anchors(
            query, dense_positions, dense_scores)
        if not anchors:
            output = list(dense_positions)[:top_k]
            return output, [1.0 / (r + 1) for r in range(len(output))], {
                "variant": variant, "fallback": "no_graph_anchor",
                "anchors": [], **anchor_diag,
                "graph_derived_in_top20": 0, "new_passages": 0,
                "seconds": perf_counter() - started,
            }
        seeds = np.zeros(self.graph.vcount(), dtype=np.float64)
        for anchor in anchors:
            seeds[anchor["node"]] = anchor["anchor_score"]
        scores, spread_diag = self.activation.run(
            query, seeds, use_hub=True, use_relation=config["use_relation"],
            hops=config["hops"])
        passage_scores = scores[self.passage_nodes]
        positive = np.flatnonzero(passage_scores > 0)
        graph_ranking = positive[np.argsort(passage_scores[positive])[::-1]].tolist()

        dense_list = list(dense_positions)
        dense_set = set(dense_list)
        merged: list[int] = []
        seen: set[int] = set()

        def add(position: int) -> None:
            if position not in seen:
                seen.add(position)
                merged.append(position)

        if self.params.merge_strategy == "interleave":
            i = j = 0
            while (i < len(dense_list) or j < len(graph_ranking)) \
                    and len(merged) < top_k:
                if i < len(dense_list):
                    add(dense_list[i]); i += 1
                if j < len(graph_ranking):
                    add(graph_ranking[j]); j += 1
        elif self.params.merge_strategy == "append":
            for position in dense_list + graph_ranking:
                add(position)
        else:
            raise ValueError(
                f"unknown merge strategy: {self.params.merge_strategy}")
        merged = merged[:top_k]
        provenance = {}
        for position in merged:
            in_dense = position in dense_set
            in_graph = position in set(graph_ranking)
            provenance[int(position)] = (
                "both" if in_dense and in_graph
                else "graph" if in_graph else "dense")
        graph_top20 = sum(
            1 for position in merged[:20]
            if provenance[int(position)] == "graph")
        return merged, [1.0 / (r + 1) for r in range(len(merged))], {
            "variant": variant, "fallback": None,
            "anchors": anchors, **anchor_diag,
            "expansion": spread_diag,
            "expanded_passages": int(len(graph_ranking)),
            "new_passages": int(len(set(graph_ranking) - dense_set)),
            "graph_derived_in_top20": int(graph_top20),
            "provenance": {str(k): v for k, v in provenance.items()},
            "seconds": perf_counter() - started,
        }


def install_graph_entry_marker(hipporag, entered: set[str]) -> None:
    """Record which queries reach ``run_ppr`` (i.e. pass the recognition gate).

    Queries never seen here fell back to dense; those are the bridge targets.
    The official result is returned unmodified.
    """
    from types import MethodType
    original_search = hipporag.graph_search_with_fact_entities
    original_run_ppr = hipporag.run_ppr
    state = {"current_query": None}

    def _search(self, query, *args, **kwargs):
        state["current_query"] = query
        try:
            return original_search(query, *args, **kwargs)
        finally:
            state["current_query"] = None

    def _run_ppr(self, reset_prob, damping=0.5):
        if state["current_query"] is not None:
            entered.add(state["current_query"])
        return original_run_ppr(reset_prob, damping)

    hipporag.graph_search_with_fact_entities = MethodType(_search, hipporag)
    hipporag.run_ppr = MethodType(_run_ppr, hipporag)
