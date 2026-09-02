"""Base graph retriever (slimmed from path_aware_retrieve.py).

KEPT: PathAwareRetriever (retrieve/retrieve_rrf/_comment_meta/...), norm_dict,
_BertTokenAdapter, label helpers, toks/STOP. DROPPED: load_query/write_markdown/
main (CLI demo). Imports rebased to .backends. Logic kept.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .backends import TfidfCosine, toks


class _BertTokenAdapter:
    """Wrap eval_retrievers.BertCosine so .scores() accepts the SAME input the
    rest of path_aware_retrieve passes to TfidfCosine (a token list). BERT wants
    raw text, so if given a token list we rejoin it. Lets the whole pipeline swap
    backends with one flag and no call-site changes."""

    name = "bert"

    def __init__(self, corpus_texts, model_name="all-MiniLM-L6-v2"):
        from .backends import BertCosine  # lazy: needs sentence-transformers
        self._be = BertCosine(corpus_texts, model_name)

    def scores(self, query):
        if isinstance(query, (list, tuple, set)):
            query = " ".join(query)
        return self._be.scores(query)


SEP = "|"
STOP = {
    "the", "and", "for", "you", "your", "with", "that", "this", "but", "not",
    "are", "was", "were", "have", "has", "had", "can", "could", "would",
    "should", "just", "like", "get", "got", "getting", "thing", "things",
    "life", "day", "time", "want", "need", "make", "made", "much", "many",
    "some", "into", "out", "all", "any", "own", "really", "very", "also",
    "then", "than", "from", "about", "because", "been", "being", "what",
    "when", "where", "there", "their", "them", "they", "will", "only",
}


def split_labels(v: object) -> list[str]:
    if v is None:
        return []
    try:
        if v != v:
            return []
    except Exception:
        pass
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    return [x.strip() for x in s.split(SEP) if x.strip()]


def clean_label(node_id: str, label: object = "") -> str:
    s = str(label or "").strip()
    if not s or s.lower() == "nan":
        s = str(node_id).split("::", 1)[-1]
    return s.replace("_", " ")


def label_tokens(label: str) -> set[str]:
    return {t for t in toks(label.replace("_", " ")) if t not in STOP and len(t) > 2}


def norm_dict(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return {k: (v - lo) / rng for k, v in d.items()}


class PathAwareRetriever:
    def __init__(self, graph_dir: Path, backend: str = "tfidf",
                 model_name: str = "all-MiniLM-L6-v2",
                 comment_text_csv: Path | None = None):
        self.graph_dir = graph_dir
        self.nodes_df = pd.read_csv(graph_dir / "graph_nodes.csv", dtype=str, keep_default_na=False)
        self.edges_df = pd.read_csv(graph_dir / "graph_edges.csv", dtype=str, keep_default_na=False)
        self.nodes = {r["node_id"]: dict(r) for _, r in self.nodes_df.iterrows()}
        self.labels = {
            nid: clean_label(nid, r.get("label", ""))
            for nid, r in self.nodes.items()
        }
        self.comment_rows = self.nodes_df[self.nodes_df["node_type"] == "comment"].copy()
        self.comment_ids = [str(x).replace("CMT::", "") for x in self.comment_rows["node_id"]]
        self.comment_text_source = "graph_nodes.text"
        self.comment_text_overrides = 0
        if comment_text_csv is not None:
            text_path = Path(comment_text_csv)
            raw = pd.read_csv(
                text_path, dtype=str, keep_default_na=False,
                usecols=["comment_id", "target_text"],
            )
            lookup = dict(zip(raw["comment_id"].astype(str), raw["target_text"].astype(str)))
            restored = []
            for cid, old in zip(self.comment_ids, self.comment_rows["text"]):
                text = lookup.get(cid, str(old or ""))
                restored.append(text)
                if cid in lookup and text != str(old or ""):
                    self.comment_text_overrides += 1
            self.comment_rows.loc[:, "text"] = restored
            for cid, text in zip(self.comment_ids, restored):
                if f"CMT::{cid}" in self.nodes:
                    self.nodes[f"CMT::{cid}"]["text"] = text
            self.comment_text_source = str(text_path)
        self.comment_texts = [str(x or "") for x in self.comment_rows["text"]]
        self.idx_of_cid = {c: i for i, c in enumerate(self.comment_ids)}
        # Semantic backend. tfidf works offline (sandbox); bert needs
        # sentence-transformers + a model download (run on your machine).
        if backend == "bert":
            self.backend = _BertTokenAdapter(self.comment_texts, model_name)
        else:
            self.backend = TfidfCosine([toks(t) for t in self.comment_texts])

        self.comment_to_post: dict[str, str] = {}
        self.post_to_facets: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self.comment_to_entities: dict[str, list[str]] = defaultdict(list)
        self.comment_to_support_edges: dict[str, list[str]] = defaultdict(list)
        self._build_edge_maps()

    def _build_edge_maps(self) -> None:
        for _, e in self.edges_df.iterrows():
            s, t, et = str(e["source_id"]), str(e["target_id"]), str(e["edge_type"])
            if et == "answered_by" and s.startswith("POST::") and t.startswith("CMT::"):
                self.comment_to_post[t.replace("CMT::", "")] = s.replace("POST::", "")
            elif et == "mentions_entity" and s.startswith("CMT::"):
                self.comment_to_entities[s.replace("CMT::", "")].append(self.labels.get(t, clean_label(t)))
            elif et == "has_support_function" and s.startswith("CMT::"):
                self.comment_to_support_edges[s.replace("CMT::", "")].append(self.labels.get(t, clean_label(t)))
            elif et in {"has_scenario", "has_need", "has_constraint"} and s.startswith("POST::"):
                key = et.replace("has_", "")
                self.post_to_facets[s.replace("POST::", "")][key].append(self.labels.get(t, clean_label(t)))

    def _query_activations(self, query: str, limit: int = 12) -> list[dict]:
        q = {t for t in toks(query) if t not in STOP and len(t) > 2}
        keep_types = {
            "support_function", "strategy_domain", "ef_mechanism",
            "scenario", "need", "constraint", "canonical_entity",
        }
        hits = []
        for nid, r in self.nodes.items():
            ntype = r.get("node_type")
            if ntype not in keep_types:
                continue
            label = self.labels[nid]
            score = len(q & label_tokens(label))
            if ntype == "canonical_entity" and score < 2:
                continue
            if score:
                hits.append({"node_id": nid, "type": ntype, "label": label, "score": score})
        hits.sort(key=lambda x: (-x["score"], x["type"], x["label"]))
        return hits[:limit]

    def _comment_meta(self, cid: str) -> dict:
        row = self.comment_rows.iloc[self.idx_of_cid[cid]].to_dict()
        post_id = self.comment_to_post.get(cid, str(row.get("post_id", "")))
        support = split_labels(row.get("support_functions")) or self.comment_to_support_edges.get(cid, [])
        domains = split_labels(row.get("strategy_domains"))
        ef = split_labels(row.get("ef_mechanisms"))
        facets = self.post_to_facets.get(post_id, {})
        entities = self.comment_to_entities.get(cid, [])
        try:
            evidence = float(row.get("evidence_strength") or 0.0)
        except Exception:
            evidence = 0.0
        return {
            "comment_id": cid,
            "post_id": post_id,
            "text": str(row.get("text") or ""),
            "support": support,
            "domains": domains,
            "ef": ef,
            "scenario": facets.get("scenario", []),
            "need": facets.get("need", []),
            "constraint": facets.get("constraint", []),
            "entities": entities,
            "evidence": evidence,
        }

    def _path_keys(self, meta: dict) -> list[tuple[str, str, str, str]]:
        supports = meta["support"] or ["unspecified support"]
        domains = meta["domains"] or ["unspecified strategy"]
        efs = meta["ef"] or ["unspecified EF"]
        scenarios = meta["scenario"] or ["unspecified scenario"]
        keys = []
        for s in supports[:3]:
            for d in domains[:3]:
                for ef in efs[:2]:
                    for sc in scenarios[:2]:
                        keys.append((clean_label("", s), clean_label("", d),
                                     clean_label("", ef), clean_label("", sc)))
        return keys[:12]

    def _load_profile(self, profile_json: str | None) -> dict[str, list[str]]:
        if not profile_json:
            return {}
        obj = json.loads(profile_json)
        out = {}
        for k, v in obj.items():
            if isinstance(v, str):
                vals = [v]
            else:
                vals = list(v or [])
            out[k] = [clean_label("", x).lower() for x in vals if str(x).strip()]
        return out

    def _profile_match_score(self, meta: dict, profile: dict[str, list[str]]) -> float:
        if not profile:
            return 0.0

        fields = {
            "support": meta["support"],
            "strategy_domain": meta["domains"],
            "domain": meta["domains"],
            "ef_mechanism": meta["ef"],
            "ef": meta["ef"],
            "scenario": meta["scenario"],
            "need": meta["need"],
            "constraint": meta["constraint"],
            "entity": meta["entities"],
            "entities": meta["entities"],
        }
        weights = {
            "support": 0.6,
            "strategy_domain": 1.2,
            "domain": 1.2,
            "ef_mechanism": 1.2,
            "ef": 1.2,
            "scenario": 0.8,
            "need": 0.6,
            "constraint": 0.7,
            "entity": 0.8,
            "entities": 0.8,
        }

        score = 0.0
        for key, wants in profile.items():
            have = [clean_label("", x).lower() for x in fields.get(key, [])]
            for w in wants:
                wtoks = label_tokens(w)
                for h in have:
                    htoks = label_tokens(h)
                    if w == h or (wtoks and wtoks <= htoks) or (wtoks & htoks):
                        score += weights.get(key, 0.5)
                        break
        return score

    def retrieve(self, query: str, top_seed: int = 100, n_paths: int = 6,
                 per_path: int = 2, profile_json: str | None = None,
                 profile_pool: int = 300) -> dict:
        profile = self._load_profile(profile_json)
        scores = self.backend.scores(toks(query))
        ranked_idx = sorted(scores, key=lambda i: -scores[i])
        seed_idx = ranked_idx[:top_seed]
        sem_raw = {self.comment_ids[i]: scores[i] for i in seed_idx}
        sem = norm_dict(sem_raw)
        qtok = set(toks(query))

        profile_scores = {}
        if profile:
            for cid in self.comment_ids:
                meta = self._comment_meta(cid)
                s = self._profile_match_score(meta, profile)
                if s > 0:
                    profile_scores[cid] = s
            for cid, _s in sorted(profile_scores.items(), key=lambda x: -x[1])[:profile_pool]:
                sem_raw.setdefault(cid, scores.get(self.idx_of_cid.get(cid, -1), 0.0))
            sem = norm_dict(sem_raw)

        path_comments: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
        for cid in sem_raw:
            meta = self._comment_meta(cid)
            entity_hit = sum(1 for e in meta["entities"] if qtok & label_tokens(e))
            facet_hit = sum(
                1 for x in meta["scenario"] + meta["need"] + meta["constraint"]
                if qtok & label_tokens(x)
            )
            graph_bonus = 0.12 * entity_hit + 0.08 * facet_hit + 0.15 * meta["evidence"]
            profile_bonus = 0.35 * profile_scores.get(cid, 0.0)
            item = {
                **meta,
                "semantic_score": round(sem.get(cid, 0.0), 4),
                "raw_semantic_score": float(sem_raw[cid]),
                "graph_bonus": round(graph_bonus, 4),
                "profile_bonus": round(profile_bonus, 4),
                "selection_score": round(sem.get(cid, 0.0) + graph_bonus + profile_bonus, 4),
            }
            for key in self._path_keys(meta):
                path_comments[key].append(item)

        path_rows = []
        for key, items in path_comments.items():
            items = sorted(items, key=lambda x: (-x["selection_score"], -x["evidence"]))
            count = len({x["comment_id"] for x in items})
            score = max(x["selection_score"] for x in items) + 0.05 * math.log1p(count)
            support, domain, ef, scenario = key
            path_rows.append({
                "path": {
                    "support_function": support,
                    "strategy_domain": domain,
                    "ef_mechanism": ef,
                    "scenario": scenario,
                },
                "score": round(score, 4),
                "candidate_count": count,
                "comments": items[:per_path],
            })

        # Diversify path list: avoid letting the same support/domain dominate.
        selected = []
        used_support = Counter()
        used_domain = Counter()
        used_comments = set()
        for row in sorted(path_rows, key=lambda x: -x["score"]):
            row["comments"] = [c for c in row["comments"] if c["comment_id"] not in used_comments]
            if not row["comments"]:
                continue
            s = row["path"]["support_function"]
            d = row["path"]["strategy_domain"]
            if used_support[s] >= 2 or used_domain[d] >= 2:
                continue
            selected.append(row)
            used_comments.update(c["comment_id"] for c in row["comments"])
            used_support[s] += 1
            used_domain[d] += 1
            if len(selected) >= n_paths:
                break
        if len(selected) < n_paths:
            seen = {json.dumps(x["path"], sort_keys=True) for x in selected}
            for row in sorted(path_rows, key=lambda x: -x["score"]):
                k = json.dumps(row["path"], sort_keys=True)
                if k in seen:
                    continue
                row["comments"] = [c for c in row["comments"] if c["comment_id"] not in used_comments]
                if not row["comments"]:
                    continue
                selected.append(row)
                used_comments.update(c["comment_id"] for c in row["comments"])
                seen.add(k)
                if len(selected) >= n_paths:
                    break

        plain_top = []
        for i in ranked_idx[:10]:
            cid = self.comment_ids[i]
            meta = self._comment_meta(cid)
            plain_top.append({
                "comment_id": cid,
                "score": float(scores[i]),
                "support": meta["support"],
                "domains": meta["domains"],
                "ef": meta["ef"],
                "text": meta["text"][:500],
            })

        return {
            "query": query,
            "structured_query_profile": profile,
            "activated_graph_labels": self._query_activations(query),
            "plain_semantic_top10": plain_top,
            "path_aware_results": selected,
        }

    # ------------------------------------------------------------------ #
    # RRF fusion + discriminative entity bridge (the "no-harm" retriever)
    # ------------------------------------------------------------------ #
    def _entity_idf(self) -> dict[str, float]:
        """IDF-style weight per entity label: rarer entity -> higher weight.
        Kills generic hubs (one entity spans up to 622 comments) that otherwise
        flood graph expansion. Cached."""
        if getattr(self, "_ent_idf_cache", None) is not None:
            return self._ent_idf_cache
        import math as _m
        ent_df: dict[str, int] = defaultdict(int)
        for cid, ents in self.comment_to_entities.items():
            for e in set(ents):
                ent_df[clean_label("", e).lower()] += 1
        N = max(len(self.comment_ids), 1)
        self._ent_idf_cache = {
            e: _m.log((N + 1) / (df + 1)) for e, df in ent_df.items()
        }
        return self._ent_idf_cache

    def retrieve_rrf(self, query: str, profile: dict | None,
                     k: int = 8, top_seed: int = 60,
                     w_sem: float = 1.0, w_graph: float = 0.2,
                     k0: int = 60, min_ent_idf: float = 1.5,
                     graph_pool: int = 200) -> list[dict]:
        """Return up to k comments by Reciprocal Rank Fusion of two rankings:

          rank_sem   : pure semantic order (the floor; always present).
          rank_graph : comments reachable via DISCRIMINATIVE entity bridges,
                       scored by sum of IDF-weighted shared-entity overlap with
                       the query/profile entities. Generic entities (low IDF)
                       are dropped (min_ent_idf), so a 622-comment hub can't
                       flood the pool.

        score(c) = w_sem/(k0+rank_sem) + w_graph/(k0+rank_graph)

        RRF makes a strong semantic hit (rank_sem=1 -> 1/(k0+1)) hard to displace,
        so the graph ADDS off-pool discriminative candidates without evicting the
        comments semantic already nailed -> fixes the relevance/grounding bleed
        while keeping the actionability gain from graph-surfaced strategies.
        """
        scores = self.backend.scores(toks(query))
        sem_ranked = sorted(scores, key=lambda i: -scores[i])
        rank_sem: dict[str, int] = {}
        for r, i in enumerate(sem_ranked, 1):
            rank_sem[self.comment_ids[i]] = r

        idf = self._entity_idf()
        # ENTITY-BRIDGE graph walk (1 hop, sparse, discriminative):
        #   seed entities = discriminative entities carried by the SEMANTIC SEED
        #   comments (top_seed) PLUS the planner profile's entities. Each seed
        #   entity is weighted by IDF (generic hubs dropped). A blind-spot comment
        #   scores by the summed IDF of seed entities it shares.
        # This bridges rephrasing: a comment the query never lexically overlaps
        # can still be reached because it shares a rare canonical entity with a
        # comment that semantic DID retrieve. (Lexical query<->entity matching
        # failed exactly because gold uses different surface words.)
        seed_ent_weight: dict[str, float] = defaultdict(float)
        for i in sem_ranked[:top_seed]:
            cid = self.comment_ids[i]
            sw = max(scores[i], 0.0)
            for e in self.comment_to_entities.get(cid, []):
                el = clean_label("", e).lower()
                w = idf.get(el, 0.0)
                if w >= min_ent_idf:
                    seed_ent_weight[el] += w * (0.2 + sw)  # weight by sem score
        if profile:
            for e in profile.get("entity", []) + profile.get("entities", []):
                el = clean_label("", e).lower()
                w = idf.get(el, 0.0)
                if w >= min_ent_idf:
                    seed_ent_weight[el] += w
                # also let profile entity TOKENS activate matching canonical ents
                ptoks = label_tokens(el)
                for cand, cidf in idf.items():
                    if cidf >= min_ent_idf and ptoks and ptoks <= label_tokens(cand):
                        seed_ent_weight[cand] += cidf

        graph_score: dict[str, float] = {}
        for cid, ents in self.comment_to_entities.items():
            s = 0.0
            for e in ents:
                el = clean_label("", e).lower()
                s += seed_ent_weight.get(el, 0.0)
            if s > 0:
                graph_score[cid] = s
        graph_ranked = sorted(graph_score, key=lambda c: -graph_score[c])[:graph_pool]
        rank_graph = {c: r for r, c in enumerate(graph_ranked, 1)}

        # candidate pool = semantic top_seed (floor) UNION graph candidates
        pool = set(self.comment_ids[i] for i in sem_ranked[:top_seed]) | set(graph_ranked)
        fused = {}
        for c in pool:
            rs = rank_sem.get(c)
            rg = rank_graph.get(c)
            sc = 0.0
            if rs is not None:
                sc += w_sem / (k0 + rs)
            if rg is not None:
                sc += w_graph / (k0 + rg)
            fused[c] = sc
        ranked = sorted(fused, key=lambda c: -fused[c])[:k]
        out = []
        for c in ranked:
            m = self._comment_meta(c)
            out.append({"comment_id": c, "text": m["text"],
                        "score": round(fused[c], 5),
                        "rank_sem": rank_sem.get(c), "rank_graph": rank_graph.get(c),
                        "support": m["support"], "domains": m["domains"], "ef": m["ef"],
                        "entities": m["entities"]})
        return out

