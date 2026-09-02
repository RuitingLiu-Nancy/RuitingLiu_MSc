"""Union-first candidate pools for fixed-graph retrieval experiments."""
from __future__ import annotations

from collections import defaultdict

from fusion.ranking import rrf_scores


def _rank_maps(rankings: dict[str, dict[str, list[str]]], qid: str):
    return {
        method: {cid: rank for rank, cid in enumerate(rows[qid], 1)}
        for method, rows in rankings.items()
    }


def _rrf_order(rankings: dict[str, dict[str, list[str]]], qid: str,
               k0: int) -> list[str]:
    runs = {
        method: [{"comment_id": cid, "score": 0.0} for cid in rows[qid]]
        for method, rows in rankings.items()
    }
    scores = rrf_scores(runs, k0=k0)
    return sorted(scores, key=lambda cid: (-scores[cid], cid))


def build_trec_style_pool(
    rankings: dict[str, dict[str, list[str]]],
    *,
    core_methods: set[str],
    novelty_gate_methods: set[str],
    new_methods: set[str],
    dense_method: str,
    bm25_method: str,
    graph_methods: set[str],
    target_size: int = 30,
    top_per_method: int = 3,
    exclusive_depth: int = 20,
    exclusive_quota: int = 2,
    disagreement_high_depth: int = 10,
    disagreement_low_depth: int = 50,
    disagreement_rank_gap: int = 20,
    disagreement_per_type: int = 1,
    rrf_k0: int = 60,
    pool_version: str = "trec-mother-pool-v1",
) -> tuple[list[dict], dict]:
    """Build a label-blind TREC-style mother pool.

    Selection uses ranks only. Raw scores, utility judgments, qrels and
    community-reference membership are deliberately absent from this API.
    """
    methods = set(rankings)
    protected_methods = core_methods | novelty_gate_methods
    for label, required in {
        "core_methods": core_methods,
        "novelty_gate_methods": novelty_gate_methods,
        "new_methods": new_methods,
        "graph_methods": graph_methods,
        "dense_method": {dense_method},
        "bm25_method": {bm25_method},
    }.items():
        missing = required - methods
        if missing:
            raise ValueError(f"{label} missing from rankings: {sorted(missing)}")
    if target_size <= 0 or top_per_method <= 0:
        raise ValueError("target_size and top_per_method must be positive")

    qid_sets = [set(rows) for rows in rankings.values()]
    if not qid_sets or any(qids != qid_sets[0] for qids in qid_sets[1:]):
        raise ValueError("all methods must contain the same query IDs")

    output: list[dict] = []
    coverage: dict[str, dict] = {}
    for qid in sorted(qid_sets[0]):
        ranks = _rank_maps(rankings, qid)
        selected: dict[str, list[str]] = defaultdict(list)
        details: dict[str, list[str]] = defaultdict(list)
        selection_order: list[str] = []

        def add(cid: str, rule: str, detail: str) -> None:
            if cid not in selected:
                selection_order.append(cid)
            if rule not in selected[cid]:
                selected[cid].append(rule)
            if detail not in details[cid]:
                details[cid].append(detail)

        # Rule 1: only pre-registered core and novelty-gated methods get top-3.
        for method in sorted(protected_methods):
            rule = "core_top3" if method in core_methods else "novelty_top3"
            for cid in rankings[method][qid][:top_per_method]:
                add(cid, rule, f"{rule}:{method}")

        # Rule 2: exclusivity is defined against the core top-20 union, not
        # against every experimental method.
        core_top = set().union(*(
            set(rankings[method][qid][:exclusive_depth])
            for method in core_methods
        ))
        for method in sorted(new_methods):
            exclusive = [
                cid for cid in rankings[method][qid][:exclusive_depth]
                if cid not in core_top
            ]
            for cid in exclusive[:exclusive_quota]:
                add(cid, "exclusive_quota", f"exclusive_quota:{method}")

        absent = 10**9
        graph_rank = {
            cid: min((ranks[m].get(cid, absent) for m in graph_methods),
                     default=absent)
            for cid in set().union(*(set(m) for m in ranks.values()))
        }
        dense_ranks = ranks[dense_method]
        bm25_ranks = ranks[bm25_method]
        rrf_order = _rrf_order(rankings, qid, rrf_k0)
        rrf_position = {cid: pos for pos, cid in enumerate(rrf_order)}

        disagreement: dict[str, list[str]] = {
            "dense_high_graph_low": [
                cid for cid, rank in dense_ranks.items()
                if rank <= disagreement_high_depth
                and graph_rank.get(cid, absent) > disagreement_low_depth
            ],
            "graph_high_dense_low": [
                cid for cid, rank in graph_rank.items()
                if rank <= disagreement_high_depth
                and dense_ranks.get(cid, absent) > disagreement_low_depth
            ],
            "bm25_high_dense_low": [
                cid for cid, rank in bm25_ranks.items()
                if rank <= disagreement_high_depth
                and dense_ranks.get(cid, absent) > disagreement_low_depth
            ],
            "shared_rank_gap": [],
        }
        for cid in rrf_order:
            present = [method_ranks[cid] for method_ranks in ranks.values()
                       if cid in method_ranks]
            if len(present) >= 2 and max(present) - min(present) >= disagreement_rank_gap:
                disagreement["shared_rank_gap"].append(cid)

        disagreement["dense_high_graph_low"].sort(
            key=lambda cid: (dense_ranks[cid], rrf_position[cid], cid))
        disagreement["graph_high_dense_low"].sort(
            key=lambda cid: (graph_rank[cid], rrf_position[cid], cid))
        disagreement["bm25_high_dense_low"].sort(
            key=lambda cid: (bm25_ranks[cid], rrf_position[cid], cid))
        disagreement["shared_rank_gap"].sort(
            key=lambda cid: (
                -(
                    max(m[cid] for m in ranks.values() if cid in m)
                    - min(m[cid] for m in ranks.values() if cid in m)
                ),
                rrf_position[cid], cid,
            ))
        for kind, candidates in disagreement.items():
            for cid in candidates[:disagreement_per_type]:
                add(cid, "disagreement", f"disagreement:{kind}")

        mandatory = len(selected)
        if mandatory > target_size:
            raise ValueError(
                f"query {qid} has {mandatory} protected/quota/disagreement "
                f"candidates but target_size={target_size}"
            )

        # Rule 4: canonical RRF fills only the remaining slots.
        for cid in rrf_order:
            if len(selected) >= target_size:
                break
            add(cid, "rrf", "rrf:k0=" + str(rrf_k0))

        # Never fabricate candidates when the union is smaller than the target.
        # The caller records the shortfall and may refuse the incomplete pool.
        selected_ids = selection_order
        for cid in selected_ids:
            rank_per_method = {
                method: method_ranks[cid]
                for method, method_ranks in ranks.items() if cid in method_ranks
            }
            methods_at_exclusive_depth = [
                method for method, rank in rank_per_method.items()
                if rank <= exclusive_depth
            ]
            output.append({
                "query_id": qid,
                "comment_id": cid,
                "selected_by_rule": selected[cid],
                "selection_details": details[cid],
                "rank_per_method": rank_per_method,
                "method_provenance": sorted(rank_per_method),
                "exclusive_methods_at_depth": (
                    sorted(methods_at_exclusive_depth)
                    if len(methods_at_exclusive_depth) == 1 else []
                ),
                "pool_version": pool_version,
                "human_gold": False,
                "llm_judged": False,
            })
        coverage[qid] = {
            "selected": len(selected_ids),
            "target_size": target_size,
            "shortfall": max(0, target_size - len(selected_ids)),
            "raw_union": len(rrf_order),
            "mandatory_before_rrf": mandatory,
            "selected_by": {
                rule: sum(rule in rules for rules in selected.values())
                for rule in ("core_top3", "novelty_top3", "exclusive_quota",
                             "disagreement", "rrf")
            },
        }
    return output, coverage


def build_candidate_pool_v2(
    rankings: dict[str, dict[str, list[str]]],
    texts: dict[str, str],
    *,
    new_methods: set[str],
    target_size: int = 30,
    top_per_method: int = 3,
    exclusive_depth: int = 20,
    exclusive_quota: int = 2,
    rrf_k0: int = 60,
    graph_entry_groups: dict[str, str] | None = None,
) -> tuple[list[dict], dict]:
    """Select a fixed union without consulting labels or utility scores."""
    qids = sorted(set.intersection(*(set(rows) for rows in rankings.values())))
    output, coverage = [], {}
    for qid in qids:
        ranks = _rank_maps(rankings, qid)
        selected: dict[str, list[str]] = defaultdict(list)
        selection_order = []

        def add(cid: str, rule: str):
            if cid not in selected:
                selection_order.append(cid)
            if rule not in selected[cid]:
                selected[cid].append(rule)

        for method, rows in rankings.items():
            for cid in rows[qid][:top_per_method]:
                add(cid, f"unconditional_top{top_per_method}:{method}")

        for method in sorted(new_methods):
            others = set().union(*(
                set(rows[qid][:exclusive_depth]) for other, rows in rankings.items()
                if other != method))
            exclusive = [cid for cid in rankings[method][qid][:exclusive_depth]
                         if cid not in others]
            for cid in exclusive[:exclusive_quota]:
                add(cid, f"new_method_top{exclusive_depth}_exclusive:{method}")

        rrf_order = _rrf_order(rankings, qid, rrf_k0)
        for cid in rrf_order:
            if len(selected) >= target_size:
                break
            add(cid, "rrf_fill")

        selected_ids = selection_order[:target_size]
        for presentation_rank, cid in enumerate(selected_ids, 1):
            provenance = {method: {"rank": method_ranks[cid]}
                          for method, method_ranks in ranks.items() if cid in method_ranks}
            top_depth_methods = [method for method, method_ranks in ranks.items()
                                 if method_ranks.get(cid, 10**9) <= exclusive_depth]
            output.append({
                "query_id": qid, "comment_id": cid, "text": texts[cid],
                "presentation_rank": presentation_rank,
                "provenance": provenance,
                "rank_per_method": {m: v["rank"] for m, v in provenance.items()},
                "exclusive_methods_at_depth": (
                    top_depth_methods if len(top_depth_methods) == 1 else []),
                "selected_by_rule": selected[cid],
                "graph_entry_group": (
                    (graph_entry_groups or {}).get(qid, "unknown")),
                "transition_profile": sorted(
                    method.removeprefix("official_") for method in provenance
                    if method.startswith("official_")),
                "pool_stage": "frozen-dev candidate-pool rehearsal",
                "human_gold": False,
            })
        coverage[qid] = {
            "selected": len(selected_ids),
            "raw_union": len(rrf_order),
            "rules": {rule: sum(rule in rules for rules in selected.values())
                      for rule in sorted({r for rules in selected.values() for r in rules})},
        }
    return output, coverage
