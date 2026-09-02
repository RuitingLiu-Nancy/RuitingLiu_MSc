"""Diversified multi-tree PCST retrieval on the Official HippoRAG2 graph.

Motivation: a single prize-collecting Steiner tree tends to pick one dominant
low-cost/high-prize region; multi-need queries may require several
complementary evidence regions.

Reuse contract: the query-local problem (anchors, bounded subgraph, prizes,
costs) and the phrase->passage mapping come verbatim from
``LocalPCSTRetriever.build_local_problem`` / ``map_selected_to_passages``.
This module only adds (a) repeated solving with reuse penalties between trees
and (b) diversity-aware merge of the trees' passages.

Provenance: sequential reuse-penalised re-solving is the standard
"peel-and-resolve" diversification heuristic (recorded as ``own`` adaptation
around the same ``pcst_fast`` core as G-Retriever); the coverage objective of
``pcst_component_coverage`` follows the facet-coverage form
``Score = Relevance + lambda*Coverage - mu*Redundancy - gamma*Cost``.
Components come from a rule-based split of the query text (NO LLM calls,
per this round's boundary); if a decomposition cache exists it can be passed
in explicitly.  Coverage/redundancy use content-token overlap only -- no
labels, no post/thread metadata, no extra embedding calls.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from candidate_pool.retrieval.local_pcst_retriever import (
    LocalPCSTRetriever,
    solve_pcst,
)


VARIANTS = {
    "pcst_multi2_node_penalty",
    "pcst_multi2_edge_penalty",
    "pcst_multi2_passage_penalty",
    "pcst_component_coverage",
}

_STOPWORDS = frozenset("""
a about after all also am an and any are as at be because been before being but
by can could did do does doing down for from had has have having he her here
hers him his how i if in into is it its just like me more most my no nor not of
off on once only or other our out over own re s so some such t than that the
their them then there these they this those through to too under until up very
was we were what when where which while who whom why will with you your
""".split())


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9']+", str(text).lower())
            if token not in _STOPWORDS and len(token) > 1}


def split_query_components(query_text: str, *, min_tokens: int = 4,
                           max_components: int = 6) -> list[str]:
    """Rule-based coarse component split (sentences, then contrast clauses)."""
    sentences = re.split(r"(?<=[.!?])\s+", str(query_text).strip())
    parts: list[str] = []
    for sentence in sentences:
        for clause in re.split(
                r",?\s+(?:but|however|also|and then|meanwhile)\s+", sentence,
                flags=re.IGNORECASE):
            clause = clause.strip()
            if len(tokenize(clause)) >= min_tokens:
                parts.append(clause)
    return parts[:max_components] if parts else [str(query_text).strip()]


def component_coverage(component: str, passage_text: str) -> float:
    component_tokens = tokenize(component)
    if not component_tokens:
        return 0.0
    return len(component_tokens & tokenize(passage_text)) / len(component_tokens)


@dataclass(frozen=True)
class MultiPCSTParameters:
    """Read from configuration/params.yaml::official_multi_pcst."""
    trees: int = 2
    node_prize_multiplier: float = 0.1     # pcst_multi2_node_penalty
    edge_cost_multiplier: float = 3.0      # pcst_multi2_edge_penalty
    passage_prize_multiplier: float = 0.1  # pcst_multi2_passage_penalty
    coverage_lambda: float = 0.5
    redundancy_mu: float = 0.5
    cost_gamma: float = 0.1
    coverage_threshold: float = 0.15
    min_component_tokens: int = 4
    max_components: int = 6
    merged_passage_cap: int = 500


class MultiPCSTRetriever:
    """Same ``retrieve`` signature as LocalPCSTRetriever -> hook-compatible."""

    def __init__(self, local: LocalPCSTRetriever, params: MultiPCSTParameters,
                 variant: str,
                 components_by_query: dict[str, list[str]] | None = None):
        if variant not in VARIANTS:
            raise ValueError(f"unknown multi-PCST variant: {variant}")
        self.local = local
        self.params = params
        self.variant = variant
        self.components_by_query = components_by_query or {}

    def _passage_text(self, position: int) -> str:
        node = int(self.local.passage_nodes[position])
        return str(self.local.graph.vs[node]["content"])

    def _apply_penalty(self, prizes: np.ndarray, costs: np.ndarray,
                       selected_local: np.ndarray,
                       selected_edges_local: np.ndarray,
                       problem) -> None:
        if self.variant == "pcst_multi2_edge_penalty":
            costs[np.asarray(selected_edges_local, dtype=np.int64)] *= (
                self.params.edge_cost_multiplier)
        elif self.variant == "pcst_multi2_passage_penalty":
            for idx in np.asarray(selected_local, dtype=np.int64):
                if problem.local_nodes[int(idx)] in self.local.passage_pos:
                    prizes[int(idx)] *= self.params.passage_prize_multiplier
        else:  # node penalty (also the diversifier for component_coverage)
            prizes[np.asarray(selected_local, dtype=np.int64)] *= (
                self.params.node_prize_multiplier)

    def _merge_interleave(self, per_tree_positions: list[list[int]]) -> list[int]:
        merged, seen = [], set()
        index = 0
        while any(index < len(positions) for positions in per_tree_positions):
            for positions in per_tree_positions:
                if index < len(positions) and positions[index] not in seen:
                    seen.add(positions[index]); merged.append(positions[index])
            index += 1
        return merged

    def _merge_coverage(self, per_tree_positions: list[list[int]],
                        components: list[str], problem,
                        tree_costs: list[float]) -> tuple[list[int], dict]:
        """Greedy: Score = Relevance + lambda*Coverage - mu*Redundancy - gamma*Cost."""
        pool: dict[int, int] = {}
        for tree_index, positions in enumerate(per_tree_positions):
            for position in positions:
                pool.setdefault(position, tree_index)
        max_cost = max(tree_costs) or 1.0
        texts = {position: self._passage_text(position) for position in pool}
        relevance = {
            position: float(problem.query_scores[
                self.local.passage_nodes[position]]) for position in pool}
        selected: list[int] = []
        covered: set[int] = set()
        coverage_map: dict[int, list[int]] = {i: [] for i in range(len(components))}
        remaining = dict(pool)
        while remaining and len(selected) < self.params.merged_passage_cap:
            best_position, best_score, best_new = None, -np.inf, set()
            for position, tree_index in remaining.items():
                new_components = {
                    i for i, component in enumerate(components)
                    if i not in covered and component_coverage(
                        component, texts[position]) >= self.params.coverage_threshold}
                redundancy = max((
                    len(tokenize(texts[position]) & tokenize(texts[other]))
                    / max(len(tokenize(texts[position])), 1)
                    for other in selected), default=0.0)
                score = (relevance[position]
                         + self.params.coverage_lambda * len(new_components)
                         - self.params.redundancy_mu * redundancy
                         - self.params.cost_gamma * tree_costs[tree_index] / max_cost)
                if score > best_score:
                    best_position, best_score, best_new = position, score, new_components
            selected.append(best_position)
            covered |= best_new
            for component_index in best_new:
                coverage_map[component_index].append(int(best_position))
            del remaining[best_position]
        return selected, {
            "components": components,
            "covered_components": len(covered),
            "component_coverage_map": {
                components[i]: positions
                for i, positions in coverage_map.items() if positions},
        }

    def retrieve(self, query: str, reset_prob: np.ndarray,
                 official_ids: np.ndarray, official_scores: np.ndarray,
                 top_k: int) -> tuple[np.ndarray, np.ndarray, dict]:
        started = perf_counter()
        problem = self.local.build_local_problem(query, reset_prob)
        if problem.fallback is not None:
            return (np.asarray(official_ids[:top_k]),
                    np.asarray(official_scores[:top_k]), {
                        "variant": self.variant, "fallback": problem.fallback,
                        "seconds": perf_counter() - started, "trees": []})
        prizes = problem.prizes.copy()
        costs = problem.costs.copy()
        trees, per_tree_positions, tree_costs = [], [], []
        for tree_index in range(self.params.trees):
            selected_local, selected_edges_local = solve_pcst(
                problem.edge_array, prizes, costs)
            nodes_global = {problem.local_nodes[int(i)] for i in selected_local}
            positions, mapping_diag = self.local.map_selected_to_passages(
                nodes_global, problem)
            edge_set = {int(e) for e in selected_edges_local}
            trees.append({
                "tree": tree_index + 1,
                "nodes": len(nodes_global),
                "edges": len(edge_set),
                "passages": len(positions),
                "node_set": nodes_global,
                "edge_set": edge_set,
                "passage_set": set(positions),
                **mapping_diag,
            })
            per_tree_positions.append(positions)
            tree_costs.append(float(
                costs[np.asarray(selected_edges_local, dtype=np.int64)].sum())
                if len(selected_edges_local) else 0.0)
            if tree_index + 1 < self.params.trees:
                self._apply_penalty(
                    prizes, costs, selected_local, selected_edges_local, problem)

        overlaps = []
        for i in range(len(trees)):
            for j in range(i + 1, len(trees)):
                overlaps.append({
                    "trees": [i + 1, j + 1],
                    "node_overlap": len(trees[i]["node_set"] & trees[j]["node_set"]),
                    "edge_overlap": len(trees[i]["edge_set"] & trees[j]["edge_set"]),
                    "passage_overlap": len(
                        trees[i]["passage_set"] & trees[j]["passage_set"]),
                })

        components = self.components_by_query.get(query) or split_query_components(
            query, min_tokens=self.params.min_component_tokens,
            max_components=self.params.max_components)
        coverage_diag: dict = {}
        if self.variant == "pcst_component_coverage":
            merged, coverage_diag = self._merge_coverage(
                per_tree_positions, components, problem, tree_costs)
        else:
            merged = self._merge_interleave(per_tree_positions)

        def coverage_count(positions: list[int]) -> int:
            return sum(
                1 for component in components
                if any(component_coverage(component, self._passage_text(p))
                       >= self.params.coverage_threshold for p in positions))

        output = list(dict.fromkeys(
            merged + np.asarray(official_ids).tolist()))[:top_k]
        scores = np.asarray(
            [1.0 / (rank + 1) for rank in range(len(output))], dtype=float)
        for tree in trees:  # sets are diagnostics-only; strip before JSON
            tree.pop("node_set"); tree.pop("edge_set"); tree.pop("passage_set")
        return np.asarray(output, dtype=np.int64), scores, {
            "variant": self.variant, "fallback": None,
            "seconds": perf_counter() - started,
            "trees": trees, "tree_costs": tree_costs,
            "tree_overlaps": overlaps,
            "components": components,
            "component_coverage_single_tree": coverage_count(
                per_tree_positions[0][:20]),
            "component_coverage_merged": coverage_count(merged[:20]),
            **coverage_diag,
            "merged_passages": len(merged),
            "official_fill": max(0, len(output) - len(merged)),
        }


def install_multi_pcst(hipporag, retrievers: dict[str, MultiPCSTRetriever],
                       sink: dict[str, dict], *, top_k: int) -> dict:
    """Run every variant on the same captured official restart vector."""
    from types import MethodType
    original_search = hipporag.graph_search_with_fact_entities
    original_run_ppr = hipporag.run_ppr
    runtime = {"graph_search_queries": 0, "multi_pcst_queries": 0,
               "seconds": 0.0, "current_query": None,
               "variants": sorted(retrievers)}

    def _search(self, query, *args, **kwargs):
        runtime["graph_search_queries"] += 1
        runtime["current_query"] = query
        try:
            return original_search(query, *args, **kwargs)
        finally:
            runtime["current_query"] = None

    def _run_ppr(self, reset_prob, damping=0.5):
        official_ids, official_scores = original_run_ppr(reset_prob, damping)
        query = runtime["current_query"]
        if query is None:
            return official_ids, official_scores
        query_sink = sink.setdefault(query, {})
        for variant, retriever in retrievers.items():
            ids, scores, diagnostics = retriever.retrieve(
                query, reset_prob, official_ids, official_scores, top_k)
            runtime["seconds"] += float(diagnostics["seconds"])
            query_sink[variant] = {"doc_ids": ids, "doc_scores": scores,
                                   "diagnostics": diagnostics}
        runtime["multi_pcst_queries"] += 1
        return official_ids, official_scores

    hipporag.graph_search_with_fact_entities = MethodType(_search, hipporag)
    hipporag.run_ppr = MethodType(_run_ppr, hipporag)
    return runtime
