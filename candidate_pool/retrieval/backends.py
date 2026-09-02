"""Retrieval backends + IR metrics (slimmed from eval_retrievers.py).

KEPT: toks, BM25, TfidfCosine, BertCosine, ndcg_at, eval_ranking, load_corpus,
load_heldout_queries (+ BaselineOverlap). DROPPED: run_pure_mode/run_graph_mode/
run_rerank_mode/PPREngine/run_ppr_mode/run_ppr_recall_mode/main (old experiments,
only called from the removed main()). Logic of kept code unchanged.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

WORD = re.compile(r"[a-z]{3,}")


def toks(text: object) -> list[str]:
    return WORD.findall(str(text).lower())


# --------------------------------------------------------------------------- #
# Backends: each returns, for a query, a dict {doc_idx: score} over the corpus.
# --------------------------------------------------------------------------- #
class BaselineOverlap:
    """Raw token-set intersection size — exactly what step4 does."""
    name = "baseline"

    def __init__(self, corpus_tokens):
        self.corpus = [set(t) for t in corpus_tokens]

    def scores(self, q_tokens):
        q = set(q_tokens)
        return {i: float(len(q & c)) for i, c in enumerate(self.corpus) if (q & c)}


class BM25:
    """Okapi BM25. Uses rank_bm25 if present, else a compact built-in."""
    name = "bm25"

    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.doclen = [len(d) for d in corpus_tokens]
        self.avgdl = sum(self.doclen) / max(self.N, 1)
        df = defaultdict(int)
        self.tf = []
        for d in corpus_tokens:
            tf = defaultdict(int)
            for w in d:
                tf[w] += 1
            self.tf.append(tf)
            for w in tf:
                df[w] += 1
        self.idf = {
            w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()
        }
        # inverted index for speed
        self.inv = defaultdict(list)
        for i, tf in enumerate(self.tf):
            for w in tf:
                self.inv[w].append(i)

    def scores(self, q_tokens):
        out = defaultdict(float)
        seen = set(q_tokens)
        for w in seen:
            if w not in self.idf:
                continue
            idf = self.idf[w]
            for i in self.inv.get(w, ()):
                tf = self.tf[i][w]
                denom = tf + self.k1 * (1 - self.b + self.b * self.doclen[i] / self.avgdl)
                out[i] += idf * (tf * (self.k1 + 1)) / denom
        return dict(out)


class TfidfCosine:
    """TF-IDF cosine. Built-in (no sklearn needed)."""
    name = "tfidf"

    def __init__(self, corpus_tokens):
        self.N = len(corpus_tokens)
        df = defaultdict(int)
        self.tf = []
        for d in corpus_tokens:
            tf = defaultdict(int)
            for w in d:
                tf[w] += 1
            self.tf.append(tf)
            for w in tf:
                df[w] += 1
        self.idf = {w: math.log((self.N + 1) / (n + 1)) + 1 for w, n in df.items()}
        # precompute doc vectors (sparse) + norms
        self.vec = []
        self.norm = []
        self.inv = defaultdict(list)
        for i, tf in enumerate(self.tf):
            v = {w: (1 + math.log(c)) * self.idf.get(w, 0.0) for w, c in tf.items()}
            self.vec.append(v)
            self.norm.append(math.sqrt(sum(x * x for x in v.values())) or 1.0)
            for w in v:
                self.inv[w].append(i)

    def scores(self, q_tokens):
        qtf = defaultdict(int)
        for w in q_tokens:
            qtf[w] += 1
        qv = {w: (1 + math.log(c)) * self.idf.get(w, 0.0) for w, c in qtf.items()}
        qnorm = math.sqrt(sum(x * x for x in qv.values())) or 1.0
        out = defaultdict(float)
        for w, qx in qv.items():
            for i in self.inv.get(w, ()):
                out[i] += qx * self.vec[i].get(w, 0.0)
        return {i: s / (self.norm[i] * qnorm) for i, s in out.items()}


class BertCosine:
    """sentence-transformers embeddings, cosine. Needs the lib + a model."""
    name = "bert"

    def __init__(self, corpus_texts, model_name="all-MiniLM-L6-v2"):
        import hashlib
        import os
        from pathlib import Path
        from sentence_transformers import SentenceTransformer  # lazy
        import numpy as np
        self.np = np
        local_only = os.environ.get("EVIDENCE_PIPELINE_HF_LOCAL_ONLY", "").lower() in {"1", "true", "yes"}
        self.model = SentenceTransformer(model_name, local_files_only=local_only)
        # Evaluation repeatedly embeds the same 19k-comment corpus.  A larger
        # CPU batch is numerically identical but makes validation ablations
        # practical; callers can reduce it for memory-constrained machines.
        batch_size = int(os.environ.get("EVIDENCE_PIPELINE_BERT_BATCH_SIZE", "256"))
        # Disk caching is deliberately opt-in.  It is an evaluation-speed aid,
        # not a retrieval change: cache identity includes model and every corpus
        # text, so a rebuilt graph cannot silently reuse stale embeddings.
        cache_dir = os.environ.get("EVIDENCE_PIPELINE_BERT_CACHE_DIR", "")
        cache_path = None
        if cache_dir:
            digest = hashlib.sha256()
            digest.update(model_name.encode("utf-8"))
            for text in corpus_texts:
                digest.update(b"\0")
                digest.update(str(text).encode("utf-8", errors="replace"))
            cache_path = Path(cache_dir) / f"bert_corpus_{digest.hexdigest()[:20]}.npy"
            if cache_path.exists():
                self.emb = np.load(cache_path, allow_pickle=False)
                return
        emb = self.model.encode(corpus_texts, batch_size=batch_size,
                                show_progress_bar=True, normalize_embeddings=True)
        self.emb = np.asarray(emb)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, self.emb)

    def scores(self, q_text):
        qv = self.model.encode([q_text], normalize_embeddings=True)[0]
        sims = self.emb @ qv  # cosine (both normalised)
        return {i: float(s) for i, s in enumerate(sims)}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def dcg(rel):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rel))


def ndcg_at(ranked_ids, gold, k):
    rel = [1.0 if c in gold else 0.0 for c in ranked_ids[:k]]
    idcg = dcg(sorted([1.0] * min(len(gold), k), reverse=True))
    return (dcg(rel) / idcg) if idcg > 0 else 0.0


def eval_ranking(ranked_ids, gold, Ks=(10, 100)):
    rankpos = {c: i + 1 for i, c in enumerate(ranked_ids)}
    best = min((rankpos[c] for c in gold if c in rankpos), default=None)
    res = {"rr": (1.0 / best) if best else 0.0,
           "ndcg10": ndcg_at(ranked_ids, gold, 10)}
    for k in Ks:
        topk = set(ranked_ids[:k])
        res[f"recall{k}"] = len(gold & topk) / len(gold)
    return res


# --------------------------------------------------------------------------- #
def load_corpus(graph_dir: Path):
    nodes = pd.read_csv(graph_dir / "graph_nodes.csv")
    edges = pd.read_csv(graph_dir / "graph_edges.csv")
    cmt = nodes[nodes["node_type"] == "comment"].copy()
    cmt_ids = [str(x).replace("CMT::", "") for x in cmt["node_id"]]
    cmt_text = [str(t or "") for t in cmt["text"]]
    idx_of_cid = {c: i for i, c in enumerate(cmt_ids)}

    post_text = {}
    for _, n in nodes[nodes["node_type"] == "post"].iterrows():
        pid = str(n["node_id"]).replace("POST::", "")
        post_text[pid] = str(n.get("text") or "")

    gold = defaultdict(set)
    for _, e in edges[edges["edge_type"] == "answered_by"].iterrows():
        pid = str(e["source_id"]).replace("POST::", "")
        cid = str(e["target_id"]).replace("CMT::", "")
        gold[pid].add(cid)

    return cmt_ids, cmt_text, idx_of_cid, post_text, dict(gold)


def load_heldout_queries(queries_csv: Path, idx_of_cid: dict):
    """HELD-OUT eval source: queries are posts NOT in the graph. Returns
    (post_text, gold) keyed by held-out post_id, exactly like the same-post
    source so the three modes run unchanged.

    The query text is the held-out post body; gold is its reply comment_ids.
    We keep ONLY gold comments that still exist in the corpus (idx_of_cid),
    so a query is scorable (its answer is a retrievable candidate). With the
    IR-protocol split this should be ALL of them, but we guard anyway and warn.
    """
    import pandas as pd
    SEP = "|"
    df = pd.read_csv(queries_csv, dtype=str, keep_default_na=False)
    post_text, gold = {}, {}
    dropped_q, dropped_gold = 0, 0
    for _, r in df.iterrows():
        pid = str(r["post_id"]).strip()
        qtext = str(r["query_text"])
        glist = [c for c in str(r["gold_comment_ids"]).split(SEP) if c]
        in_corpus = {c for c in glist if c in idx_of_cid}
        dropped_gold += len(glist) - len(in_corpus)
        if not qtext.strip() or not in_corpus:
            dropped_q += 1
            continue
        post_text[pid] = qtext
        gold[pid] = in_corpus
    print(f"[heldout] queries: {len(gold)} usable "
          f"(dropped {dropped_q} with no text/gold-in-corpus; "
          f"{dropped_gold} gold replies not in corpus)")
    return post_text, gold
