#!/usr/bin/env python3
"""Graph densification + Leiden community detection (schema-free layer).

Follows the standard GraphRAG multi-edge recipe (Edge et al. 2024; KG-RAG
surveys): an entity graph built from a SINGLE edge type is sparse. Densify it
with three standard edge types, then run community detection:

  1. RELATION edges    : LLM-judged semantic relations (already in the graph:
                         helps_with / causes_or_worsens / ...).
  2. CO-OCCURRENCE edges: entities mentioned in the SAME comment are linked,
                         weighted by co-occurrence frequency (GraphRAG default).
  3. KNN SIMILARITY edges: each entity linked to its top-k most semantically
                         similar entities (cosine over entity-label embeddings),
                         above a threshold. (TF-IDF in sandbox; BERT on machine.)

Community detection = graspologic.partition.hierarchical_leiden  ← the SAME
function Microsoft GraphRAG calls. max_cluster_size=10 = GraphRAG default.

Run:
  STUDY_RETR_BACKEND=bert python densify_and_community.py \
    --graph out/resample_graph/graph_full_4omini_t072 \
    --cooc-min 1 --knn-k 5 --knn-thresh 0.5 \
    --out out/resample_graph/communities_full_4omini.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

REL_TYPES = {"helps_with", "causes_or_worsens", "addresses_barrier",
             "contrasts_with", "is_example_of", "occurs_in_context",
             "used_for", "co_occurs_with"}


def _table_rows(graph_dir, stem):
    """Read a full graph table, preferring CSV but supporting canonical parquet.

    ``write_table`` writes Parquet plus a short CSV preview whenever a Parquet
    engine is available.  Historical graph directories also contain manually
    materialised full CSV files.  Reading both formats here keeps graph
    assembly reproducible from the builder's direct outputs.
    """
    graph_dir = Path(graph_dir)
    csv_path = graph_dir / f"{stem}.csv"
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)
        return
    parquet_path = graph_dir / f"{stem}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"missing full {stem} table: expected {csv_path} or {parquet_path}"
        )
    import pandas as pd

    yield from pd.read_parquet(parquet_path).fillna("").to_dict("records")


def load_edges(graph_dir):
    """Return (relation_edges, comment_to_entities, entity_labels)."""
    rel = []
    ment = defaultdict(set)        # comment_id -> {ENT::..}
    labels = {}
    for r in _table_rows(graph_dir, "graph_nodes"):
        if r.get("node_type") == "canonical_entity":
            labels[r["node_id"]] = r.get("label", r["node_id"])
    for r in _table_rows(graph_dir, "graph_edges"):
        et = r.get("edge_type")
        s, t = r["source_id"], r["target_id"]
        if et in REL_TYPES and s.startswith("ENT::") and t.startswith("ENT::"):
            raw_w = _float(r.get("weight"), _float(r.get("n_instances"), 1.0))
            rel.append((s, t, et, raw_w))
        elif et == "mentions_entity" and s.startswith("CMT::") and t.startswith("ENT::"):
            ment[s].add(t)
    return rel, ment, labels


def build_dense_graph(rel, ment, labels, cooc_min, knn_k, knn_thresh, backend,
                      normalize, relation_weight, cooc_weight, knn_weight):
    g = nx.Graph()
    for e in labels:
        g.add_node(e)

    raw_edges = []

    # (1) relation edges: preserve original graph weight/n_instances.
    for s, t, rel_type, raw_w in rel:
        if s != t:
            raw_edges.append((s, t, "relation", raw_w, rel_type))

    # (2) co-occurrence edges (same comment), raw weight = frequency.
    cooc = Counter()
    for cid, ents in ment.items():
        ents = sorted(ents)
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                cooc[(ents[i], ents[j])] += 1
    for (a, b), f in cooc.items():
        if f >= cooc_min:
            raw_edges.append((a, b, "cooc", float(f), "same_comment"))

    # (3) KNN semantic-similarity edges over entity labels, raw weight = cosine.
    ents = list(labels)
    texts = [_clean(labels[e]) for e in ents]
    sims = _knn(texts, knn_k, knn_thresh, backend)   # list of (i, j, score)
    for i, j, sc in sims:
        raw_edges.append((ents[i], ents[j], "knn", float(sc), "label_similarity"))

    edge_rows = normalize_edges(raw_edges, normalize, {
        "relation": relation_weight,
        "cooc": cooc_weight,
        "knn": knn_weight,
    })
    for r in edge_rows:
        _bump(g, r["source"], r["target"], r["kind"], w=r["weight"],
              raw_w=r["raw_weight"], relation=r["relation"])

    counts = Counter(r["kind"] for r in edge_rows)
    return g, dict(counts), edge_rows


def normalize_edges(raw_edges, mode, type_weights):
    """Scale relation/cooc/knn into a comparable weighted graph.

    Raw units differ:
      relation = original graph support count/weight
      cooc     = same-comment co-occurrence count
      knn      = cosine similarity

    The default `logmax` uses log1p for count-like relation/cooc, then divides
    by the per-kind maximum. KNN cosine is already bounded but is still divided
    by the observed per-kind max. Finally, each kind gets a type multiplier.
    """
    transformed = []
    for s, t, kind, raw_w, relation in raw_edges:
        raw_w = max(float(raw_w), 0.0)
        if mode in {"logmax", "log"} and kind in {"relation", "cooc"}:
            base = math.log1p(raw_w)
        else:
            base = raw_w
        transformed.append((s, t, kind, raw_w, relation, base))

    max_by_kind = defaultdict(float)
    for _, _, kind, _, _, base in transformed:
        max_by_kind[kind] = max(max_by_kind[kind], base)

    rows = []
    for s, t, kind, raw_w, relation, base in transformed:
        if mode in {"none", "raw"}:
            norm = base
        else:
            denom = max_by_kind[kind] or 1.0
            norm = base / denom
        rows.append({
            "source": s,
            "target": t,
            "kind": kind,
            "relation": relation,
            "raw_weight": raw_w,
            "norm_weight": norm,
            "type_weight": float(type_weights.get(kind, 1.0)),
            "weight": norm * float(type_weights.get(kind, 1.0)),
        })
    return rows


def _bump(g, a, b, kind, w=1.0, raw_w=1.0, relation=""):
    if g.has_edge(a, b):
        g[a][b]["weight"] += w
        g[a][b]["kinds"].add(kind)
        g[a][b]["raw_weight_sum"] += raw_w
        g[a][b]["relations"].add(relation)
    else:
        g.add_edge(a, b, weight=w, kinds={kind}, raw_weight_sum=raw_w,
                   relations={relation})


def load_dense_edge_graph(path, labels):
    """Load a frozen densification CSV as the exact Leiden input."""
    graph = nx.Graph()
    graph.add_nodes_from(labels)
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        _bump(
            graph,
            row["source"],
            row["target"],
            row["kind"],
            w=float(row["weight"]),
            raw_w=float(row["raw_weight"]),
            relation=row["relation"],
        )
    return graph, dict(Counter(row["kind"] for row in rows)), rows


def _float(x, default):
    try:
        return float(x)
    except Exception:
        return default


def _clean(lab):
    import re
    return re.sub(r"^o?\w+?_", "", str(lab)).replace("_", " ")


def _knn(texts, k, thresh, backend):
    import numpy as np
    if k <= 0 or len(texts) < 2:
        return []
    if backend == "bert":
        from sentence_transformers import SentenceTransformer
        model = os.environ.get("STUDY_BERT_MODEL", "all-MiniLM-L6-v2")
        revision = os.environ.get("STUDY_BERT_REVISION") or None
        batch_size = int(os.environ.get("STUDY_BERT_BATCH_SIZE", "64"))
        m = SentenceTransformer(model, revision=revision)
        emb = np.asarray(m.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
            batch_size=batch_size))
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Keep TF-IDF sparse: the public-graph benchmark has >20k entities, so
        # densifying either the feature matrix or NxN cosine matrix would need
        # several GB without changing the exact-neighbour definition.
        emb = TfidfVectorizer().fit_transform(texts)
    implementation = os.environ.get(
        "STUDY_KNN_IMPLEMENTATION", "nearest_neighbors"
    )
    if implementation == "legacy_full_matrix_argpartition":
        if not isinstance(emb, np.ndarray):
            emb = emb.toarray()
            emb = emb / (
                np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
            )
        sims = emb @ emb.T
        np.fill_diagonal(sims, -1)
        out = []
        for i in range(len(texts)):
            indices = np.argpartition(
                -sims[i], min(k, len(texts) - 1)
            )[:k]
            for j in indices:
                if j > i and sims[i, j] >= thresh:
                    out.append((i, int(j), float(sims[i, j])))
        return out
    if implementation != "nearest_neighbors":
        raise ValueError(f"unsupported STUDY_KNN_IMPLEMENTATION={implementation}")
    from sklearn.neighbors import NearestNeighbors
    neighbours = min(int(k) + 1, len(texts))  # +1 for the item itself
    nn = NearestNeighbors(
        n_neighbors=neighbours, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(emb)
    distances, indices = nn.kneighbors(emb, return_distance=True)
    out = []
    for i in range(len(texts)):
        kept = 0
        for distance, j in zip(distances[i], indices[i], strict=True):
            j = int(j)
            if j == i:
                continue
            similarity = 1.0 - float(distance)
            if similarity >= thresh and j > i:
                out.append((i, j, similarity))
            kept += 1
            if kept >= k:
                break
    return out


def run_leiden(g, max_cluster_size, seed):
    from graspologic.partition import hierarchical_leiden  # GraphRAG's function
    # run per connected component; assign each its own community namespace
    final = {}
    comp_id = 0
    levels = 0
    for comp in nx.connected_components(g):
        sub = g.subgraph(comp).copy()
        if sub.number_of_nodes() < 2:
            for n in sub:
                final[n] = f"c{comp_id}"
            comp_id += 1
            continue
        try:
            clusters = hierarchical_leiden(sub, max_cluster_size=max_cluster_size,
                                           random_seed=seed)
        except Exception:
            for n in sub:
                final[n] = f"c{comp_id}"
            comp_id += 1
            continue
        for c in clusters:
            levels = max(levels, c.level)
            if c.is_final_cluster:
                final[c.node] = f"k{comp_id}_{c.cluster}"
        comp_id += 1
    return final, levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--cooc-min", type=int, default=1)
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--knn-thresh", type=float, default=0.5)
    ap.add_argument("--normalize", default="logmax",
                    choices=["logmax", "max", "none", "raw"],
                    help="How to scale relation/cooc/knn weights before Leiden.")
    ap.add_argument("--relation-weight", type=float, default=1.0,
                    help="Type multiplier for LLM relation edges.")
    ap.add_argument("--cooc-weight", type=float, default=0.7,
                    help="Type multiplier for same-comment co-occurrence edges.")
    ap.add_argument("--knn-weight", type=float, default=0.5,
                    help="Type multiplier for semantic KNN similarity edges.")
    ap.add_argument("--edge-out", default=None,
                    help="Optional CSV audit file for dense weighted entity edges.")
    ap.add_argument(
        "--dense-edge-input",
        default=None,
        help=(
            "Use a frozen dense-edge CSV as the exact Leiden input instead of "
            "recomputing BERT/co-occurrence edges."
        ),
    )
    ap.add_argument("--max-cluster-size", type=int, default=10)  # GraphRAG default
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--edges-only", action="store_true",
                    help="write/audit dense edges without running Leiden")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    backend = os.environ.get("STUDY_RETR_BACKEND", "tfidf")

    rel, ment, labels = load_edges(a.graph)
    print(f"[load] {len(labels)} entities, {len(rel)} relation edges, "
          f"{len(ment)} comments with mentions")

    # baseline (relation-only) connectivity
    g0 = nx.Graph()
    for e in labels:
        g0.add_node(e)
    for s, t, _et, _w in rel:
        if s != t:
            g0.add_edge(s, t)
    cc0 = max((len(c) for c in nx.connected_components(g0)), default=0)
    print(f"[before densify] edges={g0.number_of_edges()}, "
          f"components={nx.number_connected_components(g0)}, giant={cc0} "
          f"({cc0/len(labels)*100:.0f}% of entities)")

    if a.dense_edge_input:
        g, counts, edge_rows = load_dense_edge_graph(
            a.dense_edge_input, labels
        )
    else:
        g, counts, edge_rows = build_dense_graph(
            rel, ment, labels, a.cooc_min, a.knn_k, a.knn_thresh, backend,
            a.normalize, a.relation_weight, a.cooc_weight, a.knn_weight,
        )
    cc1 = max((len(c) for c in nx.connected_components(g)), default=0)
    print(f"[after densify] edges by type: {counts}")
    print(f"[after densify] total edges={g.number_of_edges()}, "
          f"components={nx.number_connected_components(g)}, giant={cc1} "
          f"({cc1/len(labels)*100:.0f}% of entities)")
    print(f"[weights] normalize={a.normalize}; type weights: "
          f"relation={a.relation_weight}, cooc={a.cooc_weight}, knn={a.knn_weight}")

    final, levels = ({}, -1) if a.edges_only else run_leiden(
        g, a.max_cluster_size, a.seed)
    n_comm = len(set(final.values()))
    sizes = Counter(final.values())
    ss = sorted(sizes.values(), reverse=True)
    if not a.edges_only:
        print(f"\n[leiden] communities={n_comm}, levels={levels+1}, "
              f"sizes max={ss[0]} median={ss[len(ss)//2]} min={ss[-1]}")
        print(f"[leiden] entities assigned to a community: {len(final)}/{len(labels)} "
              f"({len(final)/len(labels)*100:.0f}%)")

    # sample communities for readability check
    comm_ents = defaultdict(list)
    for n, c in final.items():
        comm_ents[c].append(_clean(labels.get(n, n)))
    if not a.edges_only:
        print("\n[sample] emergent communities (no human labels):")
        for c, _ in sizes.most_common(4):
            print(f"  {c} ({sizes[c]}): " + ", ".join(comm_ents[c][:7]))

    if a.out:
        json.dump({"entity_community": final,
                   "edge_counts": counts, "n_communities": n_comm,
                   "params": vars(a), "backend": backend,
                   "weighting": {
                       "normalize": a.normalize,
                       "relation_weight": a.relation_weight,
                       "cooc_weight": a.cooc_weight,
                       "knn_weight": a.knn_weight,
                   }},
                  open(a.out, "w"), ensure_ascii=False, indent=2)
        print(f"\n[saved] {a.out}")
    if a.edge_out:
        with open(a.edge_out, "w", newline="", encoding="utf-8") as f:
            cols = ["source", "target", "kind", "relation", "raw_weight",
                    "norm_weight", "type_weight", "weight"]
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            wr.writerows(edge_rows)
        print(f"[saved] {a.edge_out}")


if __name__ == "__main__":
    main()
