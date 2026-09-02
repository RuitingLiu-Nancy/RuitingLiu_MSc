"""Multi-hop (k-hop) graph retrieval over the entity+relation layer.

Method (SG-RAG subgraph retrieval + Think-on-Graph path traversal):
  1. ENTITY LINKING  : link the query to SEED entity nodes (query tokens hit
     canonical_entity labels).
  2. k-HOP TRAVERSAL : from seeds, walk entity--relation-->entity edges up to
     `hops` steps (BFS), collecting a reasoning subgraph of entities + the typed
     relation paths between them. Each visited entity carries a decaying weight
     (1 / (1 + hop_distance)) — closer entities matter more (flow-with-decay).
  3. EVIDENCE GATHER : for every entity in the subgraph, pull the comments that
     `mentions_entity` it; score a comment by the summed decayed weight of the
     subgraph entities it mentions, plus a small bonus for lying on a relation
     PATH (not just a seed) — this is where multi-hop beats 1-hop: it surfaces
     comments reachable only via a chain like
        query→"procrastination"→[causes]→"uni difficulty"→comment.
  4. (optional) serialise the subgraph + paths as structured context for the LLM.

References:
  - SG-RAG MOT (subgraph retrieval, BFS ordering of triplets), MDPI MAKE 2025.
  - Think-on-Graph (iterative path traversal from query entities), 2024.
  - Graph RAG survey (ACM TOIS 2025): k-hop neighbourhood / path retrieval.

This module reads the SAME PathAwareRetriever the other arms use, so it needs no
extra data loading. It is additive: pure-semantic recall is always kept as a
floor (no-harm), and the multi-hop subgraph ADDS chain-reachable evidence.
"""
from __future__ import annotations

from collections import defaultdict, deque
import math
import os

# token helpers reused from the retriever module
try:
    from .base_retriever import toks, STOP  # type: ignore
except Exception:  # pragma: no cover - resolved at runtime via sys.path
    toks = None
    STOP = set()


class MultiHopRetriever:
    """Wraps a loaded PathAwareRetriever and adds k-hop entity-path retrieval.

    Reuses the retriever's nodes/edges. Builds entity adjacency + mention maps
    once, lazily.
    """

    # entity-type prefix -> short category gloss (for the 'described' seed encoder)
    _TYPE_GLOSS = {
        "odiff": "difficulty", "otask": "life task", "ostrategy": "coping strategy",
        "otool": "tool", "omed": "medication or treatment", "ocontext": "context",
        "oaffect": "feeling", "oterm": "community term", "oresource": "resource",
        "oconcept": "concept",
    }

    def __init__(self, retr):
        self.retr = retr
        self._built = False
        self._sem_ready = False
        self._sem_failed = False
        self._encoder = None          # shared ConceptEncoder (described-seed path)
        self._encoder_mode = None

    # ---- one-time indices over the entity/relation layer ----
    def _build(self):
        if self._built:
            return
        retr = self.retr
        self.ent_label = {}          # ENT::id -> label text
        self.adj = defaultdict(list)  # ENT::id -> [(relation, ENT::id), ...] both dirs
        self.ent_to_comments = defaultdict(list)  # ENT::id -> [comment_id,...]

        for nid, row in retr.nodes.items():
            if str(row.get("node_type")) == "canonical_entity":
                lab = str(row.get("label") or nid)
                # strip the o<type>_ prefix for matching (oconcept_/odiff_/oterm_…)
                self.ent_label[nid] = lab

        REL_TYPES = {"helps_with", "causes_or_worsens", "addresses_barrier",
                     "contrasts_with", "is_example_of", "occurs_in_context",
                     "used_for", "co_occurs_with"}
        for _, e in retr.edges_df.iterrows():
            s, t, et = str(e["source_id"]), str(e["target_id"]), str(e["edge_type"])
            if et in REL_TYPES and s.startswith("ENT::") and t.startswith("ENT::"):
                self.adj[s].append((et, t))
                self.adj[t].append((et + "_inv", s))   # allow reverse traversal
            elif et == "mentions_entity" and s.startswith("CMT::") and t.startswith("ENT::"):
                self.ent_to_comments[t].append(s.replace("CMT::", ""))

        # token index over entity labels for entity linking
        self.ent_tokens = {}
        for nid, lab in self.ent_label.items():
            words = set(_label_tokens(lab))
            if words:
                self.ent_tokens[nid] = words
        # degree index (for hub detection: high-degree entities cause drift)
        self.degree = {e: len(nbrs) for e, nbrs in self.adj.items()}
        self._built = True

    def _semantic_model_name(self) -> str:
        try:
            import configuration as config
            return os.environ.get(
                "EVIDENCE_PIPELINE_BERT_MODEL",
                config.params("models", "semantic_backend", default="all-MiniLM-L6-v2"),
            )
        except Exception:
            return os.environ.get("EVIDENCE_PIPELINE_BERT_MODEL", "all-MiniLM-L6-v2")

    def _ensure_semantic(self) -> bool:
        """Lazy sentence-transformer entity embeddings.

        If sentence-transformers/model files are unavailable, callers fall back
        to token overlap. This keeps the graph route usable in lightweight
        sandbox/eval environments.
        """
        if self._sem_ready:
            return True
        if self._sem_failed:
            return False
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self._sem_np = np
            local_only = os.environ.get("EVIDENCE_PIPELINE_HF_LOCAL_ONLY", "").lower() in {"1", "true", "yes"}
            self._sem_model = SentenceTransformer(
                self._semantic_model_name(), local_files_only=local_only)
            self._sem_ent_ids = list(self.ent_label.keys())
            labels = [_clean(self.ent_label[e]) for e in self._sem_ent_ids]
            self._sem_emb = np.asarray(self._sem_model.encode(
                labels, normalize_embeddings=True, show_progress_bar=False))
            self._sem_idx = {e: i for i, e in enumerate(self._sem_ent_ids)}
            self._sem_query_cache = {}
            self._sem_ready = True
            return True
        except Exception:
            self._sem_failed = True
            return False

    def _seed_encoder_mode(self) -> str:
        try:
            import configuration as config
            return str(config.params("multihop", "seed_encoder", default="bare")).lower()
        except Exception:
            return "bare"

    def _ensure_encoder(self, mode: str):
        """Lazy shared ConceptEncoder over entities, built once per mode.
        Used only when seed_encoder='described' (the ablation switch)."""
        if self._encoder is not None and self._encoder_mode == mode:
            return self._encoder
        try:
            from .concept_encoder import ConceptEncoder
            descriptions = {}
            if mode == "described":
                descriptions = self._entity_descriptions()
            self._encoder = ConceptEncoder.from_entities(
                self.retr, target_mode=mode, descriptions=descriptions)
            self._encoder_mode = mode
            return self._encoder
        except Exception:
            return None

    def _entity_descriptions(self) -> dict:
        """Build {ENT::id -> description} for the 'described' seed encoder.

        Priority:
          1) canonical cluster CSV (multihop.entity_desc_csv): uses top_phrases
             (all surface forms of the entity) + cluster_name — the richest,
             comes straight from our Agglomerative canonicalization (§1.1).
          2) fallback: entity-type category gloss (odiff -> 'difficulty', ...).
        """
        desc = {}
        try:
            import configuration as config
            csv_path = config.params("multihop", "entity_desc_csv", default="")
        except Exception:
            csv_path = ""
        if csv_path and os.path.exists(csv_path):
            try:
                import csv as _csv
                with open(csv_path, encoding="utf-8") as fh:
                    for row in _csv.DictReader(fh):
                        cid = str(row.get("canonical_id", "")).strip()
                        if not cid:
                            continue
                        phrases = str(row.get("top_phrases", "")).strip()
                        name = str(row.get("cluster_name", "")).strip()
                        gloss = (phrases or name).replace("|", " ").strip()
                        if gloss:
                            desc[f"ENT::{cid}"] = gloss
            except Exception:
                pass
        # type-gloss fallback for any entity without a CSV description
        for nid in self.ent_label:
            if nid not in desc:
                etype = str(self.retr.nodes.get(nid, {}).get("entity_type", "")).lower()
                g = self._TYPE_GLOSS.get(etype, "")
                if g:
                    desc[nid] = g
        return desc

    def _semantic_scores(self, query: str, ent_ids: list[str]) -> dict[str, float]:
        if not query or not ent_ids:
            return {}
        # described-seed path: route through the shared ConceptEncoder (label+gloss)
        mode = self._seed_encoder_mode()
        if mode == "described":
            enc = self._ensure_encoder(mode)
            if enc is not None:
                full = enc.score(query)            # {ENT::id: cosine} over all entities
                return {e: full[e] for e in ent_ids if e in full}
        # default (bare) path: original sentence-transformer over bare labels
        if not self._ensure_semantic():
            return {}
        if query not in self._sem_query_cache:
            self._sem_query_cache[query] = self._sem_np.asarray(
                self._sem_model.encode([query], normalize_embeddings=True))[0]
        qv = self._sem_query_cache[query]
        out = {}
        for e in ent_ids:
            i = self._sem_idx.get(e)
            if i is not None:
                out[e] = float(self._sem_emb[i] @ qv)
        return out

    # ---- 1. entity linking: query -> seed entities ----
    def link_seeds(self, query: str, max_seeds: int = 8) -> list[str]:
        self._build()
        q = set(_label_tokens(query))
        literal = {}
        for nid, words in self.ent_tokens.items():
            overlap = len(q & words)
            if overlap:
                # prefer specific (short-label) entities with higher overlap
                literal[nid] = overlap / (len(words) ** 0.5)

        try:
            import configuration as config
            use_semantic = bool(config.params("multihop", "semantic_seeds", default=True))
            sem_topk = int(config.params("multihop", "semantic_seed_topk", default=max_seeds))
            sem_min = float(config.params("multihop", "semantic_seed_min_sim", default=0.25))
            union_cap = int(config.params("multihop", "seed_union_cap",
                                          default=max_seeds + sem_topk))
        except Exception:
            use_semantic, sem_topk, sem_min, union_cap = True, max_seeds, 0.25, max_seeds * 2

        semantic = {}
        semantic_available = False
        if use_semantic and sem_topk > 0:
            sims = self._semantic_scores(query, list(self.ent_tokens.keys()))
            semantic_available = bool(sims)
            semantic = dict(sorted(
                ((e, s) for e, s in sims.items() if s >= sem_min),
                key=lambda x: -x[1])[:sem_topk])
        if not semantic_available:
            union_cap = max_seeds

        # Union, not intersection: preserve exact hooks while adding semantic entry
        # points for queries whose wording misses the ontology labels.
        candidates = set(literal) | set(semantic)
        scored = []
        for nid in candidates:
            scored.append((max(literal.get(nid, 0.0), semantic.get(nid, 0.0)), nid))
        scored.sort(reverse=True)
        return [nid for _, nid in scored[:union_cap]]

    def _ent_rel(self, q_tokens: set, ent: str) -> float:
        """Relevance(entity, query) = Jaccard of label tokens (backend-free).
        Used to keep only the top-b relevant neighbours per hop (beam)."""
        et = self.ent_tokens.get(ent, set())
        if not et or not q_tokens:
            return 0.0
        u = q_tokens | et
        return len(q_tokens & et) / len(u) if u else 0.0

    # ---- 2. k-hop traversal: BEAM + HUB-STOP (depth shallow, width limited) ----
    def traverse(self, seeds: list[str], hops: int = 2, max_nodes: int = 60,
                 query: str | None = None, beam: int = 0, hub_deg: int = 0,
                 query_gate: bool = False, gate_floor: float = 0.0,
                 gate_power: float = 1.0):
        """Frontier-by-frontier BFS.

        beam>0   : at each hop keep only the top-`beam` MOST query-relevant
                   neighbours per frontier (limits width -> stops the 8->172 blow-up;
                   cf. ToG/StepChain: shallow + per-step pruning beats deep漫灌).
        hub_deg>0: do NOT expand THROUGH entities whose degree >= hub_deg (they are
                   ADHD hub concepts like burnout/procrastination that route to
                   everything -> the main drift source). Hubs still收证据 as nodes,
                   they just aren't used as traversal way-stations.
        query_gate: optionally scale a reached entity's activation by its
                    query-to-entity semantic relevance.  Beam alone is a hard
                    filter: all survivors previously received the same hop
                    weight.  The gate makes propagation query-aware while
                    keeping the default behaviour exactly backwards-compatible.
        beam=0 & hub_deg=0 & query_gate=False reproduces the old无限-width BFS.
        """
        q_tokens = set(_label_tokens(query)) if (query and (beam or query_gate)) else set()
        weight = {s: 1.0 for s in seeds}
        paths = []
        seen = set()
        frontier = list(seeds)          # process hop-by-hop (BFS levels)
        for d in range(hops):
            nxt_cand = []               # (relevance, decayed_w, node, rel, nbr)
            for node in frontier:
                if node in seen:
                    continue
                seen.add(node)
                if len(seen) >= max_nodes:
                    break
                # hub-stop: don't expand THROUGH a hub (still kept as a node)
                if hub_deg and self.degree.get(node, 0) >= hub_deg and node not in seeds:
                    continue
                for rel, nbr in self.adj.get(node, []):
                    nxt_cand.append((0.0, 1.0 / (1.0 + (d + 1)), node, rel, nbr))
            # Beam is a hard query-relevance filter.  With query_gate, use the
            # same relevance score as a soft propagation gate as well.
            if beam or query_gate:
                if query:
                    sem = self._semantic_scores(query, list({x[4] for x in nxt_cand}))
                else:
                    sem = {}
                nxt_cand = [
                    (sem.get(nbr, self._ent_rel(q_tokens, nbr)), w, node, rel, nbr)
                    for _r, w, node, rel, nbr in nxt_cand
                ]
            if beam:
                nxt_cand.sort(key=lambda x: -x[0])
                nxt_cand = nxt_cand[:beam]
            new_frontier = []
            for rel_score, w, node, rel, nbr in nxt_cand:
                if query_gate:
                    # Map cosine/Jaccard relevance to [0, 1] above a tunable
                    # floor.  This preserves path breadth but prevents a merely
                    # top-b yet weakly related neighbour from scoring like a
                    # strongly aligned one.
                    denom = max(1e-9, 1.0 - gate_floor)
                    gate = max(0.0, (rel_score - gate_floor) / denom)
                    w *= gate ** max(0.0, gate_power)
                paths.append((node, rel, nbr))
                weight[nbr] = max(weight.get(nbr, 0.0), w)
                if nbr not in seen:
                    new_frontier.append(nbr)
            frontier = new_frontier
            if not frontier:
                break
        return weight, paths

    # ---- 3. gather + score evidence comments ----
    def retrieve(self, query: str, k: int = 8, hops: int = 2,
                 max_seeds: int = 8, beam: int | None = None,
                 hub_deg: int | None = None) -> list[dict]:
        self._build()
        max_nodes = 60
        query_gate = False
        gate_floor = 0.0
        gate_power = 1.0
        entity_idf_power = 0.0
        chain_bonus = 1.15
        try:
            import configuration as config
            max_seeds = config.params("multihop", "max_seeds",
                                      default=max_seeds)
            max_nodes = config.params("multihop", "max_nodes",
                                      default=max_nodes)
            # beam / hub_deg from config (0 = off -> old behaviour).
            # Explicit method args override config for experiments.
            if beam is None:
                beam = config.params("multihop", "beam", default=0) if beam is None else beam
            if hub_deg is None:
                hub_deg = config.params("multihop", "hub_deg", default=0) if hub_deg is None else hub_deg
            query_gate = bool(config.params("multihop", "query_gate", default=False))
            gate_floor = float(config.params("multihop", "gate_floor", default=0.0))
            gate_power = float(config.params("multihop", "gate_power", default=1.0))
            entity_idf_power = float(config.params("multihop", "entity_idf_power", default=0.0))
            chain_bonus = float(config.params("multihop", "chain_bonus", default=chain_bonus))
        except Exception:
            beam, hub_deg = beam or 0, hub_deg or 0
        seeds = self.link_seeds(query, max_seeds)
        if not seeds:
            # no entity hook: fall back to pure semantic (no-harm floor)
            return _semantic(self.retr, query, k)
        weight, paths = self.traverse(seeds, hops=hops, max_nodes=max_nodes,
                                      query=query, beam=beam, hub_deg=hub_deg,
                                      query_gate=query_gate, gate_floor=gate_floor,
                                      gate_power=gate_power)

        # comment score = sum over mentioned subgraph entities of their weight
        comment_score = defaultdict(float)
        comment_hops = {}
        comment_contrib = defaultdict(list)
        seed_set = set(seeds)
        # Entity DF is the appropriate specificity proxy for evidence gathering:
        # broadly-mentioned labels otherwise dominate merely by touching many
        # comments.  power=0 keeps the historical unweighted scorer.
        n_comments = max(1, len(self.retr.comment_ids))
        for ent, w in weight.items():
            is_chain = ent not in seed_set   # reached via a relation hop
            df = len(self.ent_to_comments.get(ent, []))
            ent_idf = math.log((n_comments + 1) / (df + 1)) + 1.0
            specificity = ent_idf ** max(0.0, entity_idf_power)
            for cid in self.ent_to_comments.get(ent, []):
                contribution = w * specificity * (chain_bonus if is_chain else 1.0)
                comment_score[cid] += contribution
                comment_contrib[cid].append((contribution, ent))
                comment_hops[cid] = min(comment_hops.get(cid, 9), 0 if not is_chain else 1)

        # First-discovery parent links provide a deterministic, auditable path
        # from a seed to each reached entity.  This is retrieval provenance,
        # not an LLM-generated chain-of-thought, and does not affect ranking.
        parent = {}
        for src, rel, tgt in paths:
            if tgt not in seed_set and tgt not in parent:
                parent[tgt] = (src, rel)

        ranked = sorted(comment_score.items(), key=lambda x: -x[1])[:k]
        out = []
        for cid, sc in ranked:
            meta = self.retr._comment_meta(cid) if hasattr(self.retr, "_comment_meta") else {}
            trace = None
            if comment_contrib.get(cid):
                ranked_contrib = sorted(comment_contrib[cid], key=lambda x: -x[0])
                contribution, ent = ranked_contrib[0]
                chain_contrib = [x for x in ranked_contrib if x[1] not in seed_set]
                chain_score, chain_ent = chain_contrib[0] if chain_contrib else (0.0, None)
                trace = {
                    "seed_entities": [_clean(self.ent_label.get(s, s)) for s in seeds[:8]],
                    "matched_entity_id": ent,
                    "matched_entity": _clean(self.ent_label.get(ent, ent)),
                    "path": _reconstruct_path(ent, parent, seed_set, self.ent_label),
                    "score_contribution": round(float(contribution), 6),
                    "top_contributors": [
                        {
                            "entity_id": contrib_ent,
                            "entity": _clean(self.ent_label.get(contrib_ent, contrib_ent)),
                            "is_seed": contrib_ent in seed_set,
                            "score_contribution": round(float(contrib_score), 6),
                        }
                        for contrib_score, contrib_ent in ranked_contrib[:5]
                    ],
                    "best_chain_entity_id": chain_ent,
                    "best_chain_entity": _clean(self.ent_label.get(chain_ent, chain_ent))
                    if chain_ent else "",
                    "best_chain_path": _reconstruct_path(
                        chain_ent, parent, seed_set, self.ent_label) if chain_ent else [],
                    "best_chain_contribution": round(float(chain_score), 6),
                }
            out.append({
                "comment_id": cid,
                "text": meta.get("text", ""),
                "score": round(float(sc), 4),
                "via_multihop": comment_hops.get(cid, 0) == 1,
                "domains": meta.get("domains", []),
                "ef": meta.get("ef", []),
                "support": meta.get("support", []),
                "entities": meta.get("entities", []),
                "graph_trace": trace,
            })
        # no-harm: blend in semantic floor if multi-hop returned too few
        if len(out) < k:
            have = {c["comment_id"] for c in out}
            for c in _semantic(self.retr, query, k):
                if c["comment_id"] not in have:
                    out.append(c)
                if len(out) >= k:
                    break
        return out[:k]

    def retrieve_ppr(self, query: str, k: int = 8) -> list[dict]:
        """HippoRAG-style Personalized PageRank baseline on the existing KG.

        This isolates HippoRAG's retrieval principle without rebuilding the
        corpus with its OpenIE stack: query-linked entities receive equal
        personalization mass, PPR propagates over typed entity relations, and
        passage/comment scores aggregate entity mass with IDF-like node
        specificity.  It is therefore an *adapted baseline*, not full HippoRAG.
        """
        activation = self.ppr_activation(query)
        if not activation["seeds"]:
            return _semantic(self.retr, query, k)
        seeds = activation["seeds"]
        ppr = activation["mass"]
        damping = activation["damping"]
        specificity_power = activation["specificity_power"]

        n_comments = max(1, len(self.retr.comment_ids))
        comment_score = defaultdict(float)
        comment_contrib = defaultdict(list)
        for ent, mass in ppr.items():
            df = len(self.ent_to_comments.get(ent, []))
            if not df:
                continue
            specificity = (
                math.log((n_comments + 1) / (df + 1)) + 1.0
            ) ** max(0.0, specificity_power)
            contribution = float(mass) * specificity
            for cid in self.ent_to_comments[ent]:
                comment_score[cid] += contribution
                comment_contrib[cid].append((contribution, ent))

        ranked = sorted(comment_score.items(), key=lambda row: (-row[1], row[0]))[:k]
        out = []
        for cid, score in ranked:
            meta = self.retr._comment_meta(cid) if hasattr(self.retr, "_comment_meta") else {}
            contributors = sorted(comment_contrib[cid], reverse=True)[:5]
            out.append({
                "comment_id": cid,
                "text": meta.get("text", ""),
                "score": round(float(score), 8),
                "via_hipporag_ppr": True,
                "domains": meta.get("domains", []),
                "ef": meta.get("ef", []),
                "support": meta.get("support", []),
                "entities": meta.get("entities", []),
                "graph_trace": {
                    "method": "hipporag_style_ppr",
                    "seed_entities": [_clean(self.ent_label.get(s, s)) for s in seeds],
                    "damping": damping,
                    "top_contributors": [
                        {
                            "entity_id": ent,
                            "entity": _clean(self.ent_label.get(ent, ent)),
                            "score_contribution": round(float(value), 8),
                        }
                        for value, ent in contributors
                    ],
                },
            })
        if len(out) < k:
            have = {row["comment_id"] for row in out}
            for row in _semantic(self.retr, query, k):
                if row["comment_id"] not in have:
                    out.append(row)
                if len(out) >= k:
                    break
        return out[:k]

    def ppr_activation(self, query: str) -> dict:
        """Return the exact entity activation used by :meth:`retrieve_ppr`.

        This is retrieval provenance for offline audits and visualisation.  It
        exposes the observed PPR state without inventing an LLM chain-of-thought.
        Keeping the computation here makes ``retrieve_ppr`` and explainability
        share one source of truth.
        """
        self._build()
        try:
            import configuration as config
            damping = float(config.params("hipporag_ppr", "damping", default=0.5))
            specificity_power = float(config.params(
                "hipporag_ppr", "node_specificity_power", default=1.0))
            max_iter = int(config.params("hipporag_ppr", "max_iter", default=100))
            tol = float(config.params("hipporag_ppr", "tol", default=1e-6))
        except Exception:
            damping, specificity_power, max_iter, tol = 0.5, 1.0, 100, 1e-6

        seeds = self.link_seeds(query)
        if not seeds:
            return {
                "seeds": [], "mass": {}, "damping": damping,
                "specificity_power": specificity_power,
                "max_iter": max_iter, "tol": tol,
            }

        if not hasattr(self, "_ppr_graph"):
            import networkx as nx
            graph = nx.Graph()
            graph.add_nodes_from(self.ent_label)
            for source, neighbours in self.adj.items():
                for _relation, target in neighbours:
                    graph.add_edge(source, target, weight=1.0)
            self._ppr_graph = graph

        import networkx as nx
        seed_mass = 1.0 / len(seeds)
        personalization = {seed: seed_mass for seed in seeds}
        ppr = nx.pagerank(
            self._ppr_graph,
            alpha=damping,
            personalization=personalization,
            weight="weight",
            max_iter=max_iter,
            tol=tol,
        )
        return {
            "seeds": seeds, "mass": ppr, "damping": damping,
            "specificity_power": specificity_power,
            "max_iter": max_iter, "tol": tol,
        }

    # ---- 4. serialise subgraph as structured context for the LLM ----
    def structured_context(self, query: str, hops: int = 2, max_paths: int = 20) -> str:
        self._build()
        seeds = self.link_seeds(query)
        if not seeds:
            return ""
        weight, paths = self.traverse(seeds, hops=hops)
        lines = ["REASONING SUBGRAPH (query -> entities -> related entities):"]
        lines.append("Seed concepts: " + ", ".join(
            _clean(self.ent_label.get(s, s)) for s in seeds[:6]))
        lines.append("\nRelation paths (community-derived):")
        shown = 0
        for src, rel, tgt in paths:
            if rel.endswith("_inv"):
                continue
            lines.append(f"  - {_clean(self.ent_label.get(src,src))} "
                         f"--{rel}--> {_clean(self.ent_label.get(tgt,tgt))}")
            shown += 1
            if shown >= max_paths:
                break
        return "\n".join(lines)


# ---------- helpers ----------
def _label_tokens(text: str) -> list[str]:
    import re
    # strip entity-type prefixes so "odiff_procrastination" -> tokens
    text = re.sub(r"\b(o?concept|o?diff|o?term|o?task|o?affect|o?tool|o?context|"
                  r"o?strategy|o?med|ent)_", " ", str(text).lower())
    words = re.findall(r"[a-z]{3,}", text)
    return [w for w in words if w not in STOP]


def _clean(label: str) -> str:
    import re
    return re.sub(r"^o?\w+?_", "", str(label)).replace("_", " ")


def _reconstruct_path(ent: str, parent: dict, seeds: set[str],
                      labels: dict[str, str], max_steps: int = 8) -> list[dict]:
    """Return the observed seed-to-entity traversal path for provenance.

    ``parent`` is populated from the traversal's first-discovery edges, so the
    trace is deterministic for a fixed query/configuration.  An empty path
    means the matched entity was itself a query seed.
    """
    if ent in seeds:
        return []
    rev = []
    cur = ent
    seen = set()
    while cur not in seeds and cur in parent and cur not in seen and len(rev) < max_steps:
        seen.add(cur)
        src, rel = parent[cur]
        rev.append({
            "source_id": src,
            "source": _clean(labels.get(src, src)),
            "relation": rel,
            "target_id": cur,
            "target": _clean(labels.get(cur, cur)),
        })
        cur = src
    rev.reverse()
    return rev


def _semantic(retr, query: str, k: int):
    # mirror generate_answers.semantic_arm shape without importing it
    scores = retr.backend.scores(toks(query) if toks else query.split())
    if isinstance(scores, dict):
        ranked = sorted(scores, key=lambda i: -scores[i])[:k]
    else:
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    out = []
    for i in ranked:
        cid = retr.comment_ids[i]
        m = retr._comment_meta(cid) if hasattr(retr, "_comment_meta") else {}
        out.append({"comment_id": cid, "text": m.get("text", ""),
                    "score": round(float(scores[i]), 4), "via_multihop": False,
                    "domains": m.get("domains", []), "ef": m.get("ef", []),
                    "support": m.get("support", []), "entities": m.get("entities", [])})
    return out
