"""Unified temporary-transition profiles for Official HippoRAG2 PPR."""
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Iterable

import numpy as np

from candidate_pool.retrieval.official_graph_adapter import OfficialGraphAdapter, PROFILES


@dataclass(frozen=True)
class TransitionParameters:
    alpha: float = 1.0
    beta: float = 1.0
    hub_gamma: float = 0.5
    epsilon: float = 0.05
    relation_aggregation: str = "max"


def install_transition_profiles(
    hipporag,
    adapter: OfficialGraphAdapter,
    profiles: Iterable[str],
    sink: dict[str, dict],
    *,
    top_k: int,
    params: TransitionParameters,
) -> dict:
    """Capture official reset vectors and rerun only PPR transition weights.

    The original official PPR result is always returned to upstream.  Alternative
    profile results are written to ``sink`` and therefore cannot alter recognition,
    seeds, dense fallback or the official result.
    """
    requested = tuple(dict.fromkeys(str(x) for x in profiles))
    unknown = set(requested) - PROFILES
    if unknown:
        raise ValueError(f"unknown transition profiles: {sorted(unknown)}")
    original_search = hipporag.graph_search_with_fact_entities
    original_run_ppr = hipporag.run_ppr
    runtime = {
        "profiles": list(requested), "graph_search_queries": 0,
        "queries_with_ppr": 0,
        "static_topk_exact": 0, "static_score_max_abs_diff": 0.0,
        "profile_seconds": {name: 0.0 for name in requested},
        "current_query": None,
    }

    def _search(self, query, *args, **kwargs):
        runtime["graph_search_queries"] += 1
        runtime["current_query"] = query
        try:
            return original_search(query, *args, **kwargs)
        finally:
            runtime["current_query"] = None

    def _run_ppr(self, reset_prob, damping=0.5):
        query = runtime["current_query"]
        official_ids, official_scores = original_run_ppr(reset_prob, damping)
        if query is None:
            return official_ids, official_scores
        runtime["queries_with_ppr"] += 1
        query_sink = sink.setdefault(query, {})

        static_weights, static_weight_diag = adapter.transition_weights(query, "static")
        check_ids, check_scores, static_ppr_diag = adapter.run_ppr(
            reset_prob, static_weights, damping)
        exact = np.array_equal(official_ids[:top_k], check_ids[:top_k])
        max_diff = float(np.max(np.abs(
            np.asarray(official_scores) - np.asarray(check_scores))))
        runtime["static_topk_exact"] += int(exact)
        runtime["static_score_max_abs_diff"] = max(
            runtime["static_score_max_abs_diff"], max_diff)
        if not exact or max_diff > 1e-15:
            raise RuntimeError(
                "static transition regression diverged from upstream: "
                f"topk_exact={exact}, max_abs_diff={max_diff}")

        for profile in requested:
            if profile == "static":
                ids, scores = np.asarray(official_ids), np.asarray(official_scores)
                weight_diag, ppr_diag = static_weight_diag, static_ppr_diag
            else:
                weights, weight_diag = adapter.transition_weights(
                    query, profile, alpha=params.alpha, beta=params.beta,
                    hub_gamma=params.hub_gamma, epsilon=params.epsilon,
                    relation_aggregation=params.relation_aggregation)
                ids, scores, ppr_diag = adapter.run_ppr(reset_prob, weights, damping)
            runtime["profile_seconds"][profile] += (
                float(weight_diag["seconds"]) + float(ppr_diag["seconds"]))
            query_sink[profile] = {
                "doc_ids": np.asarray(ids[:top_k], dtype=np.int64),
                "doc_scores": np.asarray(scores[:top_k], dtype=np.float64),
                "transition": weight_diag,
                "ppr": ppr_diag,
                "restart_nonzero": int(np.count_nonzero(reset_prob)),
                "restart_sum": float(np.asarray(reset_prob).sum()),
                "static_regression_topk_exact": exact,
                "static_regression_score_max_abs_diff": max_diff,
            }
        return official_ids, official_scores

    hipporag.graph_search_with_fact_entities = MethodType(_search, hipporag)
    hipporag.run_ppr = MethodType(_run_ppr, hipporag)
    return runtime
