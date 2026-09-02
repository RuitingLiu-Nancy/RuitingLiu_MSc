#!/usr/bin/env python3
"""Step 3 (doc sec 5): build the explicit three-tier hierarchical graph.

Tiers (doc 5.1):
  Tier 1 (coarse/global): support_function(6), situation(6), need(7)
  Tier 2 (mid/bridge):    strategy_domain(5+other), ef_mechanism(5+other),
                          epitome_mechanism(3), scenario(14), constraint(14)
  Tier 3 (fine/source):   post, comment, evidence_chunk, canonical_entity

Edges (doc 5.2):
  SUMMARY_OF 統辖 edges (the core):
    support_function -SUMMARY_OF-> strategy_domain / ef_mechanism   (T1->T2)
    strategy_domain  -SUMMARY_OF-> comment                          (T2->T3)
    ef_mechanism     -SUMMARY_OF-> comment                          (T2->T3, parallel)
    support_function -SUMMARY_OF-> comment   (when no subtype)
    situation -SUMMARY_OF-> scenario -SUMMARY_OF-> post
  structural: post-answered_by->comment, post-has_*->facet,
              comment-has_support_function->function, comment-has_empathy(level)->epitome,
              comment-mentions_entity->canonical_entity
  recommendation/abstraction: problem_profile-recommends_function/subtype->...,
              constraint-addressed_by->strategy_domain/ef_mechanism  (doc 5.2 new edge)
  entity relations + CO_OCCURS_WITH (weak)

evidence_strength (doc 5.4, RedditESS weak supervision):
  f(upvotes, op_replied, controversy) folded into edge weight.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from candidate_pool.graph_construction import schema as S
from candidate_pool.graph_construction.schema import RELATION_TYPES
from shared.io_utils import read_table, split_labels, write_table


def _relation_allowed(src_type: str, relation: str, tgt_type: str) -> bool:
    """Same gate llm_extract uses, applied here so backfilled emergent edges
    respect the (now widened) signatures. Empty allowed-set means 'any type'."""
    sig = RELATION_TYPES.get(relation)
    if sig is None:
        return False
    asrc, atgt = sig
    return (not asrc or src_type in asrc) and (not atgt or tgt_type in atgt)


# ---------------------------------------------------------------------------
# node / edge accumulators
# ---------------------------------------------------------------------------
def _has_text(v) -> bool:
    """True only for a real non-empty string. NaN/None/'' -> False.

    (Note: `str(nan or "")` == 'nan' which is truthy, hence this guard.)"""
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    s = str(v).strip()
    return bool(s) and s.lower() not in {"nan", "none"}


class Graph:
    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}
        self._edges: dict[tuple, dict] = {}

    def node(self, node_id: str, node_type: str, **attrs) -> str:
        if node_id not in self._nodes:
            self._nodes[node_id] = {"node_id": node_id, "node_type": node_type, **attrs}
        else:
            self._nodes[node_id].update({k: v for k, v in attrs.items() if v not in (None, "")})
        return node_id

    def edge(self, src: str, dst: str, etype: str, weight: float = 1.0, **attrs) -> None:
        key = (src, dst, etype)
        if key in self._edges:
            self._edges[key]["weight"] += weight
            self._edges[key]["n_instances"] = self._edges[key].get("n_instances", 1) + 1
        else:
            self._edges[key] = {
                "edge_id": f"{etype}:{src}->{dst}",
                "source_id": src, "target_id": dst, "edge_type": etype,
                "weight": weight, "n_instances": 1, **attrs,
            }

    def nodes_df(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._nodes.values()))

    def edges_df(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._edges.values()))


# ---------------------------------------------------------------------------
# evidence_strength (RedditESS weak supervision, doc 5.4)
# ---------------------------------------------------------------------------
def evidence_strength(comment_score, op_replied: bool, response_conf) -> float:
    import math
    # NaN guard: an absent CSV cell becomes float('nan') under pandas, and
    # float(nan) succeeds — so the except branch never fires and the nan then
    # poisons max()/log1p, yielding a nan evidence score that propagates into
    # step4's 0.7*evidence_quality term and makes every total score nan. Treat
    # nan (and unparseable) as the neutral default for each component.
    try:
        score = float(comment_score)
        if math.isnan(score):
            score = 0.0
    except (TypeError, ValueError):
        score = 0.0
    upvote_term = math.log1p(max(score, 0.0))                 # reception
    reciprocity = 1.0 if op_replied else 0.0                  # OP replied
    try:
        conf = float(response_conf)
        if math.isnan(conf):
            conf = 0.5
    except (TypeError, ValueError):
        conf = 0.5
    # weighted blend; bounded, monotone in each component
    return round(0.5 * conf + 0.4 * (upvote_term / (upvote_term + 1)) + 0.1 * reciprocity, 4)


# ---------------------------------------------------------------------------
def build(remapped_dir: Path, compiled_dir: Path, entities_dir: Path | None,
          out_dir: Path, entity_jsonl: Path | None = None) -> Graph:
    g = Graph()

    comments = read_table(remapped_dir, "comment_annotations_remapped")
    posts = read_table(compiled_dir, "post_problem_annotations")
    try:
        profiles = read_table(compiled_dir, "problem_profiles")
    except FileNotFoundError:
        profiles = pd.DataFrame()

    # pid -> post BODY text (emitted by jsonl_to_remapped from the input csv).
    # Optional: degrade gracefully if absent so older runs still build.
    post_text_map: dict[str, str] = {}
    try:
        _pt = read_table(remapped_dir, "post_texts")
        post_text_map = {str(r["post_id"]): str(r.get("post_text", ""))
                         for _, r in _pt.iterrows()}
    except FileNotFoundError:
        print("[step3] no post_texts table; post nodes will have no body text")

    # ---- Tier 1 fixed nodes (with anchoring citations as attrs) ----
    for fn in S.SUPPORT_FUNCTIONS:
        g.node(f"FUNC::{fn}", "support_function", label=fn,
               anchor="Cutrona&Russell1990;Isser&Gazit2025", tier=1)
    for sit in S.SITUATIONS:
        g.node(f"SIT::{sit}", "situation", label=sit, anchor="PATCHES_KCAP2025", tier=1)
    for need in S.NEED_TO_FUNCTION:
        g.node(f"NEED::{need}", "need", label=need, tier=1)

    # ---- Tier 2 fixed nodes ----
    for d in S.STRATEGY_DOMAINS + [S.STRATEGY_DOMAIN_OTHER]:
        g.node(f"DOM::{d}", "strategy_domain", label=d, anchor="Canela2017", tier=2,
               is_fallback=(d == S.STRATEGY_DOMAIN_OTHER))
    for ef in S.EF_MECHANISMS + [S.EF_MECHANISM_OTHER]:
        g.node(f"EF::{ef}", "ef_mechanism", label=ef, anchor="Barkley_BDEFS", tier=2,
               is_fallback=(ef == S.EF_MECHANISM_OTHER))
    for epi in S.EPITOME_MECHANISMS:
        g.node(f"EPI::{epi}", "epitome_mechanism", label=epi, anchor="Sharma2020_EPITOME", tier=2)

    # ---- SUMMARY_OF: support_function -> strategy_domain / ef_mechanism (T1->T2) ----
    # informational_support summarises both strategy axes (doc 5.2).
    for d in S.STRATEGY_DOMAINS + [S.STRATEGY_DOMAIN_OTHER]:
        g.edge(f"FUNC::{S.STRATEGY_TRIGGER_FUNCTION}", f"DOM::{d}", "SUMMARY_OF")
    for ef in S.EF_MECHANISMS + [S.EF_MECHANISM_OTHER]:
        g.edge(f"FUNC::{S.STRATEGY_TRIGGER_FUNCTION}", f"EF::{ef}", "SUMMARY_OF")
    # emotional/esteem summarise EPITOME mechanisms
    for fn in S.EPITOME_FUNCTIONS:
        for epi in S.EPITOME_MECHANISMS:
            g.edge(f"FUNC::{fn}", f"EPI::{epi}", "SUMMARY_OF")

    # ---- POST side: situation -> scenario -> post (T1->T2->T3) ----
    op_replied_map: dict[str, bool] = {}
    post_scenarios: dict[str, list[str]] = {}
    for _, p in posts.iterrows():
        pid = p.get("post_id")
        if pid is None:
            continue
        g.node(f"POST::{pid}", "post", label=str(pid),
               text=post_text_map.get(str(pid), ""),
               problem_summary=p.get("problem_summary"), tier=3)
        scenarios = split_labels(p.get("scenario_tags"))
        post_scenarios[pid] = scenarios
        for sc in scenarios:
            g.node(f"SCEN::{sc}", "scenario", label=sc, tier=2)
            g.edge(f"SCEN::{sc}", f"POST::{pid}", "SUMMARY_OF")
            g.edge(f"POST::{pid}", f"SCEN::{sc}", "has_scenario")
            sit = S.SITUATION_OF_SCENARIO.get(sc, "general_unspecified")
            g.node(f"SIT::{sit}", "situation", label=sit, tier=1)
            g.edge(f"SIT::{sit}", f"SCEN::{sc}", "SUMMARY_OF")
        for nd in split_labels(p.get("need_tags")):
            g.node(f"NEED::{nd}", "need", label=nd, tier=1)
            g.edge(f"POST::{pid}", f"NEED::{nd}", "has_need")
        for cn in split_labels(p.get("constraint_tags")):
            mesh = S.CONSTRAINT_MESH.get(cn, "")
            g.node(f"CON::{cn}", "constraint", label=cn, tier=2,
                   canonical_mesh_id=mesh,
                   reddit_only=(cn in S.CONSTRAINT_REDDIT_ONLY))
            g.edge(f"POST::{pid}", f"CON::{cn}", "has_constraint")

    # Post ids that actually became POST nodes above. answered_by must only be
    # created when the post node exists; otherwise a held-out post (excluded from
    # `posts`) would still get a dangling POST->CMT answered_by edge that leaks
    # the gold answer to a graph walk. (General fix: no dangling answered_by.)
    present_posts: set[str] = {str(p.get("post_id")) for _, p in posts.iterrows()
                               if p.get("post_id") is not None}

    # ---- COMMENT side: the two-level taxonomy + SUMMARY_OF down to comment ----
    for _, c in comments.iterrows():
        cid = c.get("comment_id")
        pid = c.get("post_id")
        if cid is None:
            continue
        funcs = split_labels(c.get("support_functions"))
        domains = split_labels(c.get("strategy_domains"))
        efs = split_labels(c.get("ef_mechanisms"))
        # honour fallback fields (guard against NaN, which is truthy in `or`)
        if _has_text(c.get("domain_other")):
            domains.append(S.STRATEGY_DOMAIN_OTHER)
        if _has_text(c.get("ef_other")):
            efs.append(S.EF_MECHANISM_OTHER)

        es = evidence_strength(c.get("comment_score"), False, c.get("response_confidence"))
        g.node(f"CMT::{cid}", "comment", label=str(cid), post_id=pid,
               text=c.get("comment_text"),
               support_functions=c.get("support_functions"),
               strategy_domains=c.get("strategy_domains"),
               ef_mechanisms=c.get("ef_mechanisms"),
               evidence_strength=es, tier=3)
        if pid is not None and str(pid) in present_posts:
            g.edge(f"POST::{pid}", f"CMT::{cid}", "answered_by")

        has_subtype = bool(domains or efs)
        for fn in funcs:
            g.node(f"FUNC::{fn}", "support_function", label=fn, tier=1)
            g.edge(f"CMT::{cid}", f"FUNC::{fn}", "has_support_function")
            # support_function -SUMMARY_OF-> comment when no subtype (doc 5.2)
            if not (fn == S.STRATEGY_TRIGGER_FUNCTION and has_subtype):
                g.edge(f"FUNC::{fn}", f"CMT::{cid}", "SUMMARY_OF", weight=es)

        # L2a domain -SUMMARY_OF-> comment
        for d in domains:
            g.node(f"DOM::{d}", "strategy_domain", label=d, tier=2)
            g.edge(f"DOM::{d}", f"CMT::{cid}", "SUMMARY_OF", weight=es)
        # L2b ef_mechanism -SUMMARY_OF-> comment (parallel axis)
        for ef in efs:
            g.node(f"EF::{ef}", "ef_mechanism", label=ef, tier=2)
            g.edge(f"EF::{ef}", f"CMT::{cid}", "SUMMARY_OF", weight=es)

        # EPITOME empathy edges with level (doc 5.2 has_empathy(level))
        for epi in S.EPITOME_MECHANISMS:
            lvl = c.get(f"epitome_{epi}", 0)
            try:
                lvl = int(lvl)
            except (TypeError, ValueError):
                lvl = 0
            if lvl > 0:
                g.edge(f"CMT::{cid}", f"EPI::{epi}", "has_empathy", weight=float(lvl), level=lvl)

    # ---- constraint -addressed_by-> strategy_domain / ef_mechanism (doc 5.2 NEW) ----
    # Derived: for each post, link its constraints to the subtype axes activated
    # by the comments answering that post.
    cmt_by_post: dict[str, list] = defaultdict(list)
    for _, c in comments.iterrows():
        if c.get("post_id") is not None:
            cmt_by_post[c["post_id"]].append(c)
    for _, p in posts.iterrows():
        pid = p.get("post_id")
        cons = split_labels(p.get("constraint_tags"))
        if not cons:
            continue
        axis_targets: set[str] = set()
        for c in cmt_by_post.get(pid, []):
            for d in split_labels(c.get("strategy_domains")):
                axis_targets.add(f"DOM::{d}")
            for ef in split_labels(c.get("ef_mechanisms")):
                axis_targets.add(f"EF::{ef}")
        for cn in cons:
            for tgt in axis_targets:
                g.edge(f"CON::{cn}", tgt, "addressed_by")

    # ---- problem_profile recommendation edges (doc 5.2) ----
    if not profiles.empty:
        # map profile -> its post's activated functions/subtypes
        for _, pr in profiles.iterrows():
            prof_id = pr.get("profile_id")
            pid = pr.get("post_id")
            if prof_id is None:
                continue
            g.node(f"PROF::{prof_id}", "problem_profile",
                   label=pr.get("profile_label"), tier=1)
            funcs, dom_ef = set(), set()
            for c in cmt_by_post.get(pid, []):
                funcs.update(split_labels(c.get("support_functions")))
                dom_ef.update(f"DOM::{d}" for d in split_labels(c.get("strategy_domains")))
                dom_ef.update(f"EF::{e}" for e in split_labels(c.get("ef_mechanisms")))
            for fn in funcs:
                g.edge(f"PROF::{prof_id}", f"FUNC::{fn}", "recommends_function")
            for tgt in dom_ef:
                g.edge(f"PROF::{prof_id}", tgt, "recommends_subtype")

    # ---- ENTITY side: canonical entities + mentions + relations ----
    if entities_dir is not None:
        try:
            ents = read_table(entities_dir, "entities_canonical")
        except FileNotFoundError:
            ents = pd.DataFrame()
        for _, e in ents.iterrows():
            eid = e.get("canonical_id")
            if not eid:
                continue
            g.node(f"ENT::{eid}", "canonical_entity", label=eid,
                   entity_type=e.get("type"),
                   canonical_mesh_id=e.get("canonical_mesh_id", ""), tier=3)
            cid = e.get("comment_id")
            if cid is not None and not (isinstance(cid, float) and pd.isna(cid)):
                g.edge(f"CMT::{cid}", f"ENT::{eid}", "mentions_entity")

        # entity-to-entity relations: validated (from raw JSONL) + backfilled
        # (from the emergent side-channel, kept only if they now pass the widened
        # signatures). Both resolve surface endpoints to canonical_ids via the
        # entities_canonical mapping, so they connect the same ENT:: nodes above.
        if entity_jsonl is not None:
            _add_entity_relations(g, ents, entity_jsonl)

    return g


def _norm_surface(s: object) -> str:
    return str(s).strip().lower()


def _add_entity_relations(g: Graph, ents: pd.DataFrame, entity_jsonl: Path) -> None:
    """Add ENT->ENT edges. Two sources:
      1. validated relations in <entity_jsonl> (already passed the gate at
         extraction time);
      2. backfill from <entity_jsonl stem>_emergent.jsonl: relation_type_violation
         rows that PASS the now-widened RELATION_TYPES signatures (no LLM re-run).
    Surface endpoints are resolved to canonical_ids using entities_canonical,
    keyed on (comment_id|post_id, lowercased text)."""
    if ents.empty:
        return
    # (unit-key, surface text) -> (canonical_id, type). A comment-level unit's
    # unit_id == comment_id, but the emergent side-channel only records unit_id
    # and post_id (no comment_id). So we register the same entity under BOTH its
    # comment_id and its post_id as unit-key, and at resolve time we try every
    # identifier the record carries (unit_id, comment_id, post_id).
    surf2canon: dict[tuple, tuple] = {}

    def _clean(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        s = str(v).strip()
        return s if s and s.lower() not in {"nan", "none"} else None

    for _, e in ents.iterrows():
        cid = e.get("canonical_id")
        if not cid:
            continue
        val = (str(cid), str(e.get("type", "")))
        surface = _norm_surface(e.get("text"))
        for key in (_clean(e.get("comment_id")), _clean(e.get("post_id"))):
            if key is not None:
                surf2canon[(key, surface)] = val

    def _resolve(rec: dict, surface: str):
        surface = _norm_surface(surface)
        for key in (rec.get("unit_id"), rec.get("comment_id"), rec.get("post_id")):
            k = _clean(key)
            if k is None:
                continue
            hit = surf2canon.get((k, surface))
            if hit:
                return hit
        return None

    n_valid = n_back = 0

    # per-unit map: any endpoint string the LLM used (raw text OR its canonical)
    # -> the raw `text` registered in entities_canonical. The emergent
    # side-channel records relation endpoints as the *canonical* string (not the
    # raw text), so backfill needs this to translate them before resolving.
    unit_local: dict[str, dict[str, str]] = {}

    # --- 1. validated relations from the raw extraction JSONL ---
    if entity_jsonl.exists():
        with entity_jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # build a per-unit text+canonical -> resolution shortcut so
                # relation endpoints that used the canonical string also resolve
                local: dict[str, str] = {}
                for ent in rec.get("entities", []):
                    ent_text = ent.get("text") or ent.get("surface")
                    for k in (ent_text, ent.get("canonical")):
                        if k:
                            local[_norm_surface(k)] = _norm_surface(ent_text)
                uid = rec.get("unit_id")
                if uid is not None:
                    unit_local[str(uid)] = local
                for rel in rec.get("relations", []):
                    rtype = rel.get("relation")
                    s_surf = local.get(_norm_surface(rel.get("source")), _norm_surface(rel.get("source")))
                    t_surf = local.get(_norm_surface(rel.get("target")), _norm_surface(rel.get("target")))
                    src = _resolve(rec, s_surf)
                    tgt = _resolve(rec, t_surf)
                    if not src or not tgt:
                        continue
                    g.edge(f"ENT::{src[0]}", f"ENT::{tgt[0]}", rtype,
                           relation_kind="entity_relation", backfilled=False)
                    n_valid += 1

    # --- 2. backfill from the emergent side-channel ---
    emergent = entity_jsonl.with_name(entity_jsonl.stem + "_emergent.jsonl")
    if emergent.exists():
        with emergent.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("kind") != "relation_type_violation":
                    continue
                rtype = rec.get("relation")
                s_type = rec.get("source_type", "")
                t_type = rec.get("target_type", "")
                if not _relation_allowed(s_type, rtype, t_type):
                    continue  # still illegitimate -> stays in emergent for review
                # translate canonical endpoints -> raw text via this unit's map
                local = unit_local.get(str(rec.get("unit_id")), {})
                s_surf = local.get(_norm_surface(rec.get("source")), _norm_surface(rec.get("source")))
                t_surf = local.get(_norm_surface(rec.get("target")), _norm_surface(rec.get("target")))
                src = _resolve(rec, s_surf)
                tgt = _resolve(rec, t_surf)
                if not src or not tgt:
                    continue
                g.edge(f"ENT::{src[0]}", f"ENT::{tgt[0]}", rtype,
                       relation_kind="entity_relation", backfilled=True)
                n_back += 1

    print(f"[step3] entity relations: validated={n_valid} backfilled={n_back}")


def run(remapped_dir: Path, compiled_dir: Path, entities_dir: Path | None,
        out_dir: Path, entity_jsonl: Path | None = None) -> None:
    g = build(remapped_dir, compiled_dir, entities_dir, out_dir, entity_jsonl)
    nodes, edges = g.nodes_df(), g.edges_df()
    write_table(nodes, out_dir, "graph_nodes")
    write_table(edges, out_dir, "graph_edges")

    summary = {
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "nodes_by_type": nodes["node_type"].value_counts().to_dict() if not nodes.empty else {},
        "edges_by_type": edges["edge_type"].value_counts().to_dict() if not edges.empty else {},
        "summary_of_edges": int((edges["edge_type"] == "SUMMARY_OF").sum()) if not edges.empty else 0,
        "entity_relation_edges": (
            int((edges.get("relation_kind") == "entity_relation").sum())
            if not edges.empty and "relation_kind" in edges.columns else 0),
        "backfilled_edges": (
            int((edges.get("backfilled") == True).sum())
            if not edges.empty and "backfilled" in edges.columns else 0),
        "dual_axis_present": {
            "strategy_domain_nodes": int((nodes["node_type"] == "strategy_domain").sum()) if not nodes.empty else 0,
            "ef_mechanism_nodes": int((nodes["node_type"] == "ef_mechanism").sum()) if not nodes.empty else 0,
        },
    }
    (out_dir / "graph_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[step3] nodes:", summary["n_nodes"], "edges:", summary["n_edges"])
    print("[step3] SUMMARY_OF edges:", summary["summary_of_edges"])
    print("[step3] nodes_by_type:", summary["nodes_by_type"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remapped-dir", required=True, type=Path)
    ap.add_argument("--compiled-dir", required=True, type=Path)
    ap.add_argument("--entities-dir", type=Path, default=None)
    ap.add_argument("--entity-jsonl", type=Path, default=None,
                    help="raw extraction JSONL (entities_v2_full.jsonl); its "
                         "_emergent sibling is used to backfill now-legitimate edges")
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    run(a.remapped_dir, a.compiled_dir, a.entities_dir, a.out_dir, a.entity_jsonl)


if __name__ == "__main__":
    main()
