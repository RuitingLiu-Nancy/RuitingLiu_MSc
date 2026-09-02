"""Fixed-hop query-aware spreading activation on the Official HippoRAG2 graph.

Update rule (per step)::

    a_{t+1}(v) = g(q, v) * sum_{u in N(v)} a_t(u) * w(u, v) * h(u, v)

- ``w(u,v)``  official persisted edge weight (never modified);
- ``h(u,v)``  optional symmetric degree penalty (reuses
  ``hub_correction.symmetric_degree_penalty`` via the adapter) and/or relation
  compatibility (reuses ``RelationSidecar.aggregate`` via the adapter);
- ``g(q,v)``  per-step query-node gate ``epsilon + s(q, v)`` built from the
  same official entity/passage embedding stores
  (``OfficialGraphAdapter.query_node_similarity``).

Contrast with PPR: no global convergence, no damping teleport; propagation is
cut at 2-3 hops, every step re-applies the query gate, and only the strongest
``top_k_active`` nodes stay active.  Provenance: the gated-propagation idea
follows spreading-activation retrieval (Crestani 1997 survey) and per-step
query conditioning follows this project's query-aware transition profiles;
there is no single upstream codebase, so the propagation loop is recorded as
``own`` in docs_v2/11 §F (numpy scatter over the official edge list).

The module never mutates the graph: it reads the adapter's cached edge arrays
and writes only per-query activation vectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from candidate_pool.retrieval.official_graph_adapter import OfficialGraphAdapter


VARIANTS = {
    "spread_node_2hop": {"use_hub": False, "use_relation": False, "hops": 2},
    "spread_node_hub_2hop": {"use_hub": True, "use_relation": False, "hops": 2},
    "spread_node_relation_2hop": {"use_hub": False, "use_relation": True, "hops": 2},
    "spread_node_hub_3hop": {"use_hub": True, "use_relation": False, "hops": 3},
}


@dataclass(frozen=True)
class SpreadingParameters:
    """Read from configuration/params.yaml::official_spreading."""
    top_k_active: int = 2000            # per-step active-node cap
    gate_epsilon: float = 0.05          # g(q,v) = gate_epsilon + s(q,v)
    hub_gamma: float = 0.5              # same default as transition profiles
    relation_beta: float = 1.0
    relation_epsilon: float = 0.05
    relation_aggregation: str = "max"
    early_stop_mass_ratio: float = 1e-6  # stop when step mass decays below this
    accumulate: bool = True             # score = sum of a_t over steps (incl. a_0)


def build_edge_weights(adapter: OfficialGraphAdapter, query: str, *,
                       use_hub: bool, use_relation: bool,
                       params: SpreadingParameters) -> np.ndarray:
    """Compose w*h from the adapter's canonical primitives (no duplication)."""
    profile = "hub" if use_hub else "static"
    weights, _ = adapter.transition_weights(
        query, profile, hub_gamma=params.hub_gamma)
    if use_relation:
        relation_scores = adapter.query_relation_similarity(query)
        per_edge = adapter.relation_sidecar.aggregate(
            relation_scores, params.relation_aggregation)
        mask = np.isfinite(per_edge)
        weights[mask] *= np.power(
            params.relation_epsilon + per_edge[mask], params.relation_beta)
    return weights


class SpreadingActivation:
    """Bounded propagation over the immutable official edge arrays."""

    def __init__(self, adapter: OfficialGraphAdapter, params: SpreadingParameters):
        self.adapter = adapter
        self.params = params
        self.passage_nodes = np.asarray(
            adapter.hipporag.passage_node_idxs, dtype=np.int64)

    def run(self, query: str, seeds: np.ndarray, *, use_hub: bool,
            use_relation: bool, hops: int) -> tuple[np.ndarray, dict]:
        """Return (node activation scores, step diagnostics)."""
        started = perf_counter()
        if hops < 1:
            raise ValueError("hops must be >= 1")
        n = self.adapter.graph.vcount()
        seeds = np.asarray(seeds, dtype=np.float64)
        if seeds.shape != (n,):
            raise ValueError("seed vector must cover every official node")
        seeds = np.where(np.isfinite(seeds) & (seeds > 0), seeds, 0.0)
        initial_mass = float(seeds.sum())
        if initial_mass <= 0:
            return np.zeros(n), {"steps": [], "empty_seeds": True,
                                 "seconds": perf_counter() - started}
        a = seeds / initial_mass
        gate = self.params.gate_epsilon + self.adapter.query_node_similarity(query)
        weights = build_edge_weights(
            self.adapter, query, use_hub=use_hub, use_relation=use_relation,
            params=self.params)
        src = self.adapter.edge_sources
        dst = self.adapter.edge_targets
        total = a.copy() if self.params.accumulate else None
        steps = []
        for step in range(hops):
            step_started = perf_counter()
            a_next = np.zeros(n, dtype=np.float64)
            # Undirected scatter in both directions: sum_u a(u) w(u,v) h(u,v).
            np.add.at(a_next, dst, a[src] * weights)
            np.add.at(a_next, src, a[dst] * weights)
            a_next *= gate
            propagated_mass = float(a_next.sum())
            active = int(np.count_nonzero(a_next))
            dropped_mass = 0.0
            if active > self.params.top_k_active:
                threshold = np.partition(
                    a_next, -self.params.top_k_active)[-self.params.top_k_active]
                pruned = a_next < threshold
                dropped_mass = float(a_next[pruned].sum())
                a_next[pruned] = 0.0
            kept_mass = float(a_next.sum())
            steps.append({
                "step": step + 1,
                "active_nodes_before_prune": active,
                "active_nodes_after_prune": int(np.count_nonzero(a_next)),
                "propagated_mass": propagated_mass,
                "kept_mass": kept_mass,
                "dropped_mass": dropped_mass,
                "high_degree_mass": float(
                    a_next[self.adapter.high_degree].sum()),
                "seconds": perf_counter() - step_started,
            })
            a = a_next
            if total is not None:
                total += a
            if kept_mass < self.params.early_stop_mass_ratio:
                steps[-1]["early_stopped"] = True
                break
        scores = total if total is not None else a
        return scores, {
            "steps": steps,
            "empty_seeds": False,
            "final_high_degree_mass": float(scores[self.adapter.high_degree].sum()),
            "seconds": perf_counter() - started,
        }

    def rank_passages(self, scores: np.ndarray, official_ids: np.ndarray,
                      top_k: int) -> tuple[np.ndarray, np.ndarray, dict]:
        """Ranked passage positions; short outputs filled with official order."""
        passage_scores = scores[self.passage_nodes]
        positive = np.flatnonzero(passage_scores > 0)
        order = positive[np.argsort(passage_scores[positive])[::-1]]
        output = list(dict.fromkeys(
            order.tolist() + np.asarray(official_ids).tolist()))[:top_k]
        # Rank-only score contract, same as local PCST.
        ranked_scores = np.asarray(
            [1.0 / (rank + 1) for rank in range(len(output))], dtype=float)
        return np.asarray(output, dtype=np.int64), ranked_scores, {
            "activated_passages": int(len(order)),
            "official_fill": max(0, len(output) - len(order)),
        }


def install_spreading(hipporag, activation: SpreadingActivation,
                      variants: list[str], sink: dict[str, dict], *,
                      top_k: int) -> dict:
    """Capture the official restart vector and run all variants per query.

    Same lifecycle contract as ``install_local_pcst``: the official result is
    returned unmodified, dense-fallback queries never reach the hook, and the
    graph is never written.
    """
    from types import MethodType
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown spreading variants: {sorted(unknown)}")
    original_search = hipporag.graph_search_with_fact_entities
    original_run_ppr = hipporag.run_ppr
    runtime = {"graph_search_queries": 0, "spreading_queries": 0,
               "seconds": 0.0, "current_query": None,
               "variants": list(variants)}

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
        for variant in variants:
            config = VARIANTS[variant]
            scores, diagnostics = activation.run(
                query, np.asarray(reset_prob, dtype=np.float64),
                use_hub=config["use_hub"], use_relation=config["use_relation"],
                hops=config["hops"])
            ids, ranked_scores, mapping = activation.rank_passages(
                scores, official_ids, top_k)
            runtime["seconds"] += float(diagnostics["seconds"])
            query_sink[variant] = {
                "doc_ids": ids, "doc_scores": ranked_scores,
                "diagnostics": {**diagnostics, **mapping,
                                "restart_nonzero": int(np.count_nonzero(reset_prob))},
            }
        runtime["spreading_queries"] += 1
        return official_ids, official_scores

    hipporag.graph_search_with_fact_entities = MethodType(_search, hipporag)
    hipporag.run_ppr = MethodType(_run_ppr, hipporag)
    return runtime
