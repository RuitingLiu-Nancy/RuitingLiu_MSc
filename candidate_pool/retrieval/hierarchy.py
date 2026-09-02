"""Paradigm B — ontology-hierarchy (SUMMARY_OF) top-down retrieval.

Companion to retrieval/multihop.py (paradigm A). Same graph, DIFFERENT edges:
  - multihop  : walks ENTITY RELATION edges (horizontal, local search).
  - hierarchy : walks SUMMARY_OF concept edges (vertical, hierarchical/global).

Design (GRAPH_DESIGN_NOTES.md §2.3) — leakage-free:
  ENTRY (入口):   query -> top ONTOLOGY CONCEPT nodes (support_function /
                  strategy_domain / ef_mechanism) via the shared ConceptEncoder
                  (label[+description]). We match CONCEPTS, never comments, so no
                  gold answer leaks into the entry step.
  ROUTING (路由): from each seed concept, follow SUMMARY_OF edges DOWN to the
                  comments it summarises. A comment's score =
                      Σ_over_seed_concepts  concept_relevance(query, c)
                                           * edge_weight(evidence_strength)
                                           * level_decay
                  with a per-concept width cap (top-N comments per concept) so a
                  broad concept ("informational_support") can't flood — the
                  hierarchical analogue of multihop's beam.
  Depth is bounded (concept -> comment is 1–2 SUMMARY_OF hops), so unlike the
  entity walk it does NOT drift (no 8->172 blow-up).

Entry chains (`entry`):
  "function"  : support_function / strategy_domain / ef_mechanism  (DEFAULT;
                lands on comments, evaluable under the held-out IR protocol).
  Note: the situation/scenario chain lands on POSTS, which are removed in the
  held-out split, so it is not usable for evaluation and is not offered here.

Returns ranked comments [{comment_id, score}], the SAME shape as
multihop.retrieve(), so it plugs into the retrieval ablation and the generation
pipeline unchanged.
"""
from __future__ import annotations

from collections import defaultdict

# ontology concept node types that can summarise comments (T1/T2 -> comment)
_CONCEPT_TYPES = ("support_function", "strategy_domain", "ef_mechanism")


class HierarchyRetriever:
    """Wraps a loaded PathAwareRetriever and adds SUMMARY_OF top-down retrieval."""

    def __init__(self, retr):
        self.retr = retr
        self._built = False
        self._encoder = None
        self._encoder_mode = None

    # ---- one-time index over SUMMARY_OF concept->comment edges ----
    def _build(self):
        if self._built:
            return
        retr = self.retr
        # concept node id -> [(comment_id, weight), ...]
        self.concept_to_comments = defaultdict(list)
        # concept node id -> node_type (for kind filtering / decay by level)
        self.concept_type = {}
        for nid, row in retr.nodes.items():
            if str(row.get("node_type")) in _CONCEPT_TYPES:
                self.concept_type[nid] = str(row.get("node_type"))

        for _, e in retr.edges_df.iterrows():
            if str(e.get("edge_type")) != "SUMMARY_OF":
                continue
            s, t = str(e["source_id"]), str(e["target_id"])
            # we only want CONCEPT -SUMMARY_OF-> COMMENT (T1/T2 -> T3)
            if s in self.concept_type and t.startswith("CMT::"):
                try:
                    w = float(e.get("weight") or 1.0)
                except Exception:
                    w = 1.0
                self.concept_to_comments[s].append((t.replace("CMT::", ""), w))
        self._built = True

    # ---- shared concept encoder (entry point matcher) ----
    def _ensure_encoder(self, target_mode: str):
        if self._encoder is not None and self._encoder_mode == target_mode:
            return self._encoder
        try:
            from .concept_encoder import ConceptEncoder
            # optional per-concept descriptions (e.g. schema definitions) could be
            # passed here; default uses the concept label (+ type as light gloss).
            descriptions = {}
            if target_mode == "described":
                for nid, ntype in self.concept_type.items():
                    descriptions[nid] = ntype.replace("_", " ")
            self._encoder = ConceptEncoder.from_ontology(
                self.retr, kinds=_CONCEPT_TYPES,
                target_mode=target_mode, descriptions=descriptions)
            self._encoder_mode = target_mode
            return self._encoder
        except Exception:
            return None

    def _cfg(self, key, default):
        try:
            import configuration as config
            return config.params("hierarchy", key, default=default)
        except Exception:
            return default

    # ---- 1. ENTRY: query -> seed concepts (no comment leakage) ----
    def seed_concepts(self, query: str, top_concepts: int = 6,
                      min_sim: float = 0.0, target_mode: str | None = None):
        """Return [(concept_node_id, relevance)] for the most query-relevant
        ontology concepts. Matches CONCEPTS only -> leakage-free."""
        self._build()
        target_mode = target_mode or str(self._cfg("concept_encoder", "bare"))
        enc = self._ensure_encoder(target_mode)
        if enc is None:
            return []
        return enc.top(query, k=top_concepts, min_sim=min_sim,
                       kinds=set(_CONCEPT_TYPES))

    # ---- 2. ROUTING: seed concepts -> comments via SUMMARY_OF ----
    def retrieve(self, query: str, k: int = 100, **kwargs) -> list[dict]:
        """Top-down SUMMARY_OF retrieval. Returns [{comment_id, score}] (ranked).

        kwargs (else read from config 'hierarchy'):
          top_concepts : how many seed concepts          (default 6)
          per_concept  : width cap, top-N comments/concept (default 50)
          min_sim      : min concept-query cosine to seed  (default 0.0)
          level_decay  : multiply concept relevance by this per concept (broad
                         concepts already down-weighted via per_concept cap)
        """
        self._build()
        top_concepts = int(kwargs.get("top_concepts",
                                      self._cfg("top_concepts", 6)))
        per_concept = int(kwargs.get("per_concept",
                                     self._cfg("per_concept", 50)))
        min_sim = float(kwargs.get("min_sim", self._cfg("min_sim", 0.0)))
        target_mode = kwargs.get("concept_encoder",
                                 self._cfg("concept_encoder", "bare"))

        seeds = self.seed_concepts(query, top_concepts=top_concepts,
                                   min_sim=min_sim, target_mode=target_mode)
        if not seeds:
            return []

        # IDF-style down-weight of broad concepts (summarise huge comment sets)
        score = defaultdict(float)
        for concept_id, rel in seeds:
            children = self.concept_to_comments.get(concept_id, [])
            if not children:
                continue
            df = max(len(children), 1)
            inv_df = 1.0 / (df ** 0.5)          # broad concept -> smaller per-hit weight
            # width cap: strongest-evidence comments first
            top_children = sorted(children, key=lambda cw: -cw[1])[:per_concept]
            for cid, w in top_children:
                score[cid] += float(rel) * w * inv_df

        ranked = sorted(score.items(), key=lambda x: -x[1])[:k]
        return [{"comment_id": cid, "score": s} for cid, s in ranked]
