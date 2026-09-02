#!/usr/bin/env python3
"""2 x 3 retrieval-layer comparison: structure source x retrieval route.

Question this answers
---------------------
If we have two ways to organize the same evidence comments:

  1. ontology  : manual/theory labels (support + strategy domain + EF)
  2. community : schema-free Leiden communities over open entities

can we compare them using the same retrieval metrics we already used
(Recall@10/100, MRR, nDCG@10)?

Yes, with one caveat: pure text retrieval routes (semantic and BM25) do not use
the structure, so their ontology/community rows should be identical baselines.
The meaningful structure-sensitive rows are:
  - graph_bridge: semantic floor + one structure bridge route
  - fusion_rrf  : semantic + BM25 + one structure bridge route, RRF
  - fusion_cc   : semantic + BM25 + one structure bridge route, score fusion
  - with --graph-route old_multihop, the graph route is the original full
    MultiHopRetriever from study_platform/backend/multihop_retrieve.py.

This script reuses existing project code:
  - eval_retrievers.py: BM25, TF-IDF/BERT, eval_ranking, load_heldout_queries
  - path_aware_retrieve.py: PathAwareRetriever and comment metadata
  - communities_full_*.json: entity -> Leiden community mapping

Graph-bridge retrieval
----------------------
For a query, take top semantic seed comments. Extract their structure labels:
  - ontology bridge: support/domain/EF labels from seed comments
  - community bridge: Leiden communities attached to seed-comment entities

Then retrieve comments sharing those labels/communities and fuse rankings using
RRF. This keeps the protocol parallel across the two structures.

The `fusion_rrf` and `fusion_cc` rows mirror the earlier
"semantic + BM25 + graph" ablation:
semantic and BM25 are identical in both structure conditions; only the graph
bridge differs between ontology and community.  The default fusion weights match
the old diagnostic scripts: semantic=1.0, bm25=0.5, multihop/bridge=0.3.

Run:
  python eval_structure_retrieval_2x3.py \
    --graph-dir out/resample_graph/graph_full_4omini_t072 \
    --heldout out/resample_graph/heldout_queries_strat.csv \
    --communities out/resample_graph/communities_full_4omini_bert.json \
    --semantic tfidf \
    --out out/resample_graph/structure_retrieval_2x3_tfidf

For final dense numbers, run with --semantic bert on the local machine where the
sentence-transformers model is available.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidate_pool.retrieval.backends import (
    BM25,
    BertCosine,
    TfidfCosine,
    load_heldout_queries,
    toks,
)
from candidate_pool.retrieval.base_retriever import PathAwareRetriever
from candidate_pool.retrieval.multihop import MultiHopRetriever
import configuration as config
from evaluation.ir_metrics import eval_full as ir_eval_full, mean_metrics as ir_mean


def rank_from_scores(scores: dict[int, float], cmt_ids: list[str]) -> list[str]:
    return [cmt_ids[i] for i in sorted(scores, key=lambda i: -scores[i])]


def rrf_fuse(rank_lists: dict[str, list[str]], weights: dict[str, float] | None = None,
             k0: int = 60) -> list[str]:
    weights = weights or {}
    score = defaultdict(float)
    for route, ranked in rank_lists.items():
        w = weights.get(route, 1.0)
        for r, cid in enumerate(ranked, 1):
            score[cid] += w / (k0 + r)
    return sorted(score, key=lambda c: -score[c])


def minmax(vals: dict[str, float]) -> dict[str, float]:
    if not vals:
        return {}
    lo, hi = min(vals.values()), max(vals.values())
    if hi <= lo:
        return {k: 1.0 for k in vals}
    return {k: (v - lo) / (hi - lo) for k, v in vals.items()}


def scored_run(ranked: list[str], score_by_cid: dict[str, float], k: int) -> dict[str, float]:
    return {cid: float(score_by_cid.get(cid, 0.0)) for cid in ranked[:k]}


def scored_dicts_to_run(items: list[dict], k: int) -> dict[str, float]:
    out = {}
    for c in items[:k]:
        cid = str(c.get("comment_id", c.get("cid", "")))
        if cid:
            out[cid] = float(c.get("score", 0.0))
    return out


def fuse_rrf_runs(source_runs: dict[str, dict[str, float]], weights: dict[str, float],
                  k0: int = 60) -> list[str]:
    score = defaultdict(float)
    for name, run in source_runs.items():
        w = weights.get(name, 1.0)
        for rank, cid in enumerate(run.keys(), 1):
            score[cid] += w / (k0 + rank)
    return sorted(score, key=lambda c: -score[c])


def fuse_cc_runs(source_runs: dict[str, dict[str, float]], weights: dict[str, float]) -> list[str]:
    norm = {name: minmax(run) for name, run in source_runs.items()}
    total_w = sum(weights.get(name, 0.0) for name in source_runs) or 1.0
    score = defaultdict(float)
    for name, run in norm.items():
        w = weights.get(name, 0.0) / total_w
        for cid, sc in run.items():
            score[cid] += w * sc
    return sorted(score, key=lambda c: -score[c])



def load_entity_communities(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))["entity_community"]


def load_comment_entities(graph_dir: Path) -> dict[str, set[str]]:
    out = defaultdict(set)
    with (graph_dir / "graph_edges.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("edge_type") == "mentions_entity":
                s, t = r.get("source_id", ""), r.get("target_id", "")
                if s.startswith("CMT::") and t.startswith("ENT::"):
                    out[s.replace("CMT::", "")].add(t)
    return out


def ontology_labels(meta: dict) -> set[str]:
    labs = set()
    for axis in ("support", "domains", "ef"):
        for v in meta.get(axis, []) or []:
            if v:
                labs.add(f"{axis}:{v}")
    return labs


class StructureBridge:
    """Builds ontology/community bridge rankings over comments."""

    def __init__(self, retr: PathAwareRetriever, ent2comm: dict[str, str],
                 comment_entities: dict[str, set[str]],
                 concept_of: dict[str, str] | None = None):
        self.retr = retr
        self.ent2comm = ent2comm
        self.comment_entities = comment_entities
        self.concept_of = concept_of or {}
        self.comment_ontology: dict[str, set[str]] = {}
        self.comment_community: dict[str, set[str]] = {}
        self.comment_concept: dict[str, set[str]] = {}      # concept layer (fusion)
        self.ontology_to_comments = defaultdict(set)
        self.community_to_comments = defaultdict(set)
        self.concept_to_comments = defaultdict(set)
        self._build()

    def _build(self) -> None:
        for cid in self.retr.comment_ids:
            meta = self.retr._comment_meta(cid)
            olabs = ontology_labels(meta)
            self.comment_ontology[cid] = olabs
            for lab in olabs:
                self.ontology_to_comments[lab].add(cid)

            ents = self.comment_entities.get(cid, set())
            comms = {self.ent2comm[e] for e in ents if e in self.ent2comm}
            self.comment_community[cid] = comms
            for c in comms:
                self.community_to_comments[c].add(cid)

            # concept layer = build-time fusion label per entity (theory@data)
            cons = {self.concept_of[e] for e in ents if e in self.concept_of}
            self.comment_concept[cid] = cons
            for c in cons:
                self.concept_to_comments[c].add(cid)

    def bridge_scores(self, structure: str, sem_ranked: list[str], sem_scores_by_cid: dict[str, float],
                      top_seed: int = 60) -> dict[str, float]:
        if structure == "ontology":
            c2labels = self.comment_ontology
            label2comments = self.ontology_to_comments
        elif structure == "community":
            c2labels = self.comment_community
            label2comments = self.community_to_comments
        elif structure == "concept":
            c2labels = self.comment_concept
            label2comments = self.concept_to_comments
        else:
            raise ValueError(structure)

        label_weight = defaultdict(float)
        for rank, cid in enumerate(sem_ranked[:top_seed], 1):
            # Higher semantic seed rank and higher semantic score contribute more.
            seed_w = (1.0 / rank) + max(sem_scores_by_cid.get(cid, 0.0), 0.0)
            for lab in c2labels.get(cid, set()):
                label_weight[lab] += seed_w

        comment_score = defaultdict(float)
        for lab, w in label_weight.items():
            # down-weight broad/hub labels so "informational_support" does not flood.
            df = max(len(label2comments.get(lab, ())), 1)
            inv_df = 1.0 / (df ** 0.5)
            for cid in label2comments.get(lab, ()):
                comment_score[cid] += w * inv_df

        return dict(comment_score)

    def bridge_rank(self, structure: str, sem_ranked: list[str], sem_scores_by_cid: dict[str, float],
                    top_seed: int = 60, graph_pool: int = 200) -> list[str]:
        scores = self.bridge_scores(structure, sem_ranked, sem_scores_by_cid, top_seed=top_seed)
        return sorted(scores, key=lambda cid: -scores[cid])[:graph_pool]


def run(graph_dir: Path, heldout: Path, communities: Path, semantic: str, model: str,
        out_dir: Path, top_seed: int, graph_pool: int, graph_route: str,
        hops: int, n: int = 0, drug_filter: bool = True,
        concept: Path | None = None) -> None:
    retr = PathAwareRetriever(graph_dir, backend=semantic, model_name=model)
    cmt_ids = retr.comment_ids
    cmt_text = retr.comment_texts
    idx_of_cid = retr.idx_of_cid
    post_text, gold = load_heldout_queries(heldout, idx_of_cid)
    # drug filter — keep evaluation口径 consistent with the user study (药物 query 不展示)
    if drug_filter:
        from .safety_filter import from_config as _drug
        df, _ = _drug(None)
        if df:
            before = len(gold)
            gold = {pid: g for pid, g in gold.items()
                    if pid in post_text and not df.is_drug_related(post_text[pid])}
            print(f"[2x3] drug filter: {before} -> {len(gold)} queries (excluded {before-len(gold)})")
    if n and n < len(gold):
        import itertools
        gold = dict(itertools.islice(gold.items(), n))
    print(f"[2x3] graph={graph_dir}")
    print(f"[2x3] heldout={heldout.name}; usable queries={len(gold)}; semantic={semantic}")

    corpus_tokens = [toks(t) for t in cmt_text]
    bm25 = BM25(corpus_tokens)
    if semantic == "bert":
        sem_backend = BertCosine(cmt_text, model)
        sem_score = lambda q: sem_backend.scores(q)
    else:
        sem_backend = TfidfCosine(corpus_tokens)
        sem_score = lambda q: sem_backend.scores(toks(q))

    concept_of = {}
    structures = ["ontology", "community"]
    if concept is not None:
        concept_of = json.loads(Path(concept).read_text(encoding="utf-8"))["concept_of"]
        structures.append("concept")
        print(f"[2x3] concept layer loaded: {len(concept_of)} entities -> "
              f"{len(set(concept_of.values()))} concepts")
    bridge = StructureBridge(retr, load_entity_communities(communities),
                             load_comment_entities(graph_dir), concept_of=concept_of)
    old_mh = None
    if graph_route == "old_multihop":
        if MultiHopRetriever is None:
            raise RuntimeError(
                "Cannot import study_platform/backend/multihop_retrieve.py. "
                "Expected it under ../study_platform/backend relative to this script."
            )
        old_mh = MultiHopRetriever(retr)
        old_mh._build()

    rows = []
    per_query_rows = []
    fusion_weights = config.fusion_weights()
    source_pool = max(100, top_seed, graph_pool)
    for structure in structures:
        graph_name = "old_multihop" if graph_route == "old_multihop" else "graph_bridge"
        accum = {route: [] for route in ("semantic", "bm25", graph_name, "fusion_rrf", "fusion_cc")}

        for pid, gset in gold.items():
            query = post_text[pid]
            sem_scores = sem_score(query)
            sem_ranked = rank_from_scores(sem_scores, cmt_ids)
            sem_scores_by_cid = {cmt_ids[i]: float(v) for i, v in sem_scores.items()}

            bm_scores = bm25.scores(toks(query))
            bm_ranked = rank_from_scores(bm_scores, cmt_ids)
            bm_scores_by_cid = {cmt_ids[i]: float(v) for i, v in bm_scores.items()}

            if graph_route == "old_multihop":
                assert old_mh is not None
                mh_items = old_mh.retrieve(query, k=source_pool, hops=hops)
                graph_scores = scored_dicts_to_run(mh_items, source_pool)
                graph_ranked = list(graph_scores.keys())[:graph_pool]
                graph_only_ranked = graph_ranked
            else:
                graph_scores = bridge.bridge_scores(structure, sem_ranked, sem_scores_by_cid, top_seed=top_seed)
                graph_ranked = sorted(graph_scores, key=lambda cid: -graph_scores[cid])[:graph_pool]
                graph_only_ranked = rrf_fuse(
                    {"semantic": sem_ranked[:top_seed], "bridge": graph_ranked},
                    weights={"semantic": 1.0, "bridge": 0.4},
                )
            source_runs = {
                "semantic": scored_run(sem_ranked, sem_scores_by_cid, source_pool),
                "bm25": scored_run(bm_ranked, bm_scores_by_cid, source_pool),
                "multihop": scored_run(graph_ranked, graph_scores, source_pool),
            }
            fusion_rrf = fuse_rrf_runs(source_runs, fusion_weights)
            fusion_cc = fuse_cc_runs(source_runs, fusion_weights)

            rankings = {
                "semantic": sem_ranked,
                "bm25": bm_ranked,
                graph_name: graph_only_ranked,
                "fusion_rrf": fusion_rrf,
                "fusion_cc": fusion_cc,
            }
            for route, ranked in rankings.items():
                m = ir_eval_full(ranked, gset)        # full IR metric set
                accum[route].append(m)
                per_query_rows.append({
                    "post_id": pid, "structure": structure, "route": route, **m})

        for route, vals in accum.items():
            m = ir_mean(vals)
            rows.append({"structure": structure, "route": route,
                         "n": len(vals), **m})

    # ---- BOTH GRAPHS together: 4 ways to combine ontology + community bridges ----
    # (only meaningful for structure_bridge; both bridges available per query)
    if graph_route == "structure_bridge":
        both_routes = ("both_4way_cc", "both_union_cc", "both_inter_weighted_cc",
                       "both_inter_only_cc")
        accum_b = {r: [] for r in both_routes}
        for pid, gset in gold.items():
            query = post_text[pid]
            sem_scores = sem_score(query)
            sem_ranked = rank_from_scores(sem_scores, cmt_ids)
            sem_by_cid = {cmt_ids[i]: float(v) for i, v in sem_scores.items()}
            bm_scores = bm25.scores(toks(query))
            bm_ranked = rank_from_scores(bm_scores, cmt_ids)
            bm_by_cid = {cmt_ids[i]: float(v) for i, v in bm_scores.items()}

            onto = bridge.bridge_scores("ontology", sem_ranked, sem_by_cid, top_seed=top_seed)
            comm = bridge.bridge_scores("community", sem_ranked, sem_by_cid, top_seed=top_seed)
            onto_ranked = sorted(onto, key=lambda c: -onto[c])[:graph_pool]
            comm_ranked = sorted(comm, key=lambda c: -comm[c])[:graph_pool]
            onto_set, comm_set = set(onto_ranked), set(comm_ranked)

            sem_run = scored_run(sem_ranked, sem_by_cid, source_pool)
            bm_run = scored_run(bm_ranked, bm_by_cid, source_pool)
            w = config.fusion_weights()
            gw = float(w.get("multihop", 0.3))   # graph route weight from config

            # ① 四路 CC: semantic + bm25 + ontology_bridge + community_bridge
            r_4way = fuse_cc_runs(
                {"semantic": sem_run, "bm25": bm_run,
                 "onto": scored_run(onto_ranked, onto, source_pool),
                 "comm": scored_run(comm_ranked, comm, source_pool)},
                {"semantic": w.get("semantic", 1.0), "bm25": w.get("bm25", 0.5),
                 "onto": gw, "comm": gw})

            # ② 并集 + CC: 两图召回并集成一路 graph
            union_scores = {c: max(onto.get(c, 0.0), comm.get(c, 0.0))
                            for c in (onto_set | comm_set)}
            r_union = fuse_cc_runs(
                {"semantic": sem_run, "bm25": bm_run,
                 "graph": scored_run(sorted(union_scores, key=lambda c: -union_scores[c]),
                                     union_scores, source_pool)},
                {"semantic": w.get("semantic", 1.0), "bm25": w.get("bm25", 0.5),
                 "graph": gw})

            # ③ 交集加权 CC: 不丢独有, 但两图都召回到的额外 ×1.5
            inter = onto_set & comm_set
            graph_w_scores = dict(union_scores)
            for c in inter:
                graph_w_scores[c] = graph_w_scores.get(c, 0.0) * 1.5
            r_interw = fuse_cc_runs(
                {"semantic": sem_run, "bm25": bm_run,
                 "graph": scored_run(sorted(graph_w_scores, key=lambda c: -graph_w_scores[c]),
                                     graph_w_scores, source_pool)},
                {"semantic": w.get("semantic", 1.0), "bm25": w.get("bm25", 0.5),
                 "graph": gw})

            # ④ 纯交集 CC: 只保留两图都召回到的证据 (会丢互补性, 作对照)
            inter_scores = {c: max(onto.get(c, 0.0), comm.get(c, 0.0)) for c in inter}
            r_interonly = fuse_cc_runs(
                {"semantic": sem_run, "bm25": bm_run,
                 "graph": scored_run(sorted(inter_scores, key=lambda c: -inter_scores[c]),
                                     inter_scores, source_pool)},
                {"semantic": w.get("semantic", 1.0), "bm25": w.get("bm25", 0.5),
                 "graph": gw})

            for route, ranked in (("both_4way_cc", r_4way), ("both_union_cc", r_union),
                                  ("both_inter_weighted_cc", r_interw),
                                  ("both_inter_only_cc", r_interonly)):
                m = ir_eval_full(ranked, gset)
                accum_b[route].append(m)
                per_query_rows.append({"post_id": pid, "structure": "both",
                                       "route": route, **m})
        for route, vals in accum_b.items():
            m = ir_mean(vals)
            rows.append({"structure": "both", "route": route, "n": len(vals), **m})

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "structure_retrieval_2x3_metrics.csv", index=False)
    pd.DataFrame(per_query_rows).to_csv(out_dir / "structure_retrieval_2x3_per_query.csv", index=False)
    summary = {
        "graph_dir": str(graph_dir),
        "heldout": str(heldout),
        "communities": str(communities),
        "semantic": semantic,
        "model": model,
        "top_seed": top_seed,
        "graph_pool": graph_pool,
        "graph_route": graph_route,
        "hops": hops,
        "fusion_weights": fusion_weights,
        "note": (
            "semantic and bm25 are text-only baselines and should be identical "
            "across ontology/community. With graph_route=structure_bridge, "
            "graph_bridge/fusion_rrf/fusion_cc are structure-sensitive. With "
            "graph_route=old_multihop, the graph route reuses the original full "
            "entity-relation MultiHopRetriever; ontology/community rows are "
            "identical unless you provide/build separate ontology/community "
            "graph directories."
        ),
        "rows": rows,
    }
    (out_dir / "structure_retrieval_2x3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n[2x3] wrote -> {out_dir / 'structure_retrieval_2x3_metrics.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", required=True, type=Path)
    ap.add_argument("--heldout", required=True, type=Path)
    ap.add_argument("--communities", required=True, type=Path)
    ap.add_argument("--semantic", default="tfidf", choices=["tfidf", "bert"])
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--top-seed", type=int, default=60)
    ap.add_argument("--graph-pool", type=int, default=200)
    ap.add_argument("--graph-route", default="structure_bridge",
                    choices=["structure_bridge", "old_multihop"],
                    help="structure_bridge compares ontology/community labels; "
                         "old_multihop reuses the original entity-relation k-hop retriever.")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--n", type=int, default=0, help="sample N queries (0=all)")
    ap.add_argument("--no-drug-filter", action="store_true",
                    help="keep drug-related queries (default: filter them, 165->~113)")
    ap.add_argument("--concept", type=Path, default=None,
                    help="concept_layer.json (adds a 'concept' structure = "
                         "build-time ontology+community fusion)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    run(args.graph_dir, args.heldout, args.communities, args.semantic, args.model,
        args.out, args.top_seed, args.graph_pool, args.graph_route, args.hops, args.n,
        drug_filter=not args.no_drug_filter, concept=args.concept)


if __name__ == "__main__":
    main()
