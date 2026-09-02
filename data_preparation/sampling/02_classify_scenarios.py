#!/usr/bin/env python3
"""Few-shot scenario classification of deep-hub candidates (no training, no new
labels). Uses the 2338 already-annotated posts as a reference set and assigns
each candidate a primary scenario by embedding nearest-neighbour.

Two classifiers (both reported + a confidence flag):
  prototype : each scenario -> mean embedding of its reference posts; candidate
              -> nearest prototype (robust to per-class variance, fast).
  knn       : candidate -> majority vote of its k nearest reference posts.
  confidence: 'high' if prototype and knn agree, else 'low'.

This is WEAK (silver) labelling -- only used to BUCKET candidates by scenario so
we can balance-sample which deep-hub posts to add. The chosen posts get real LLM
post-annotation later, so the noise never enters the final graph.

Backends (swap with --backend):
  tfidf : offline, sandbox-runnable, validates the whole pipeline.
  bert  : sentence-transformers all-MiniLM-L6-v2 (run on your machine for the
          real semantic classification).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

WORD = re.compile(r"[a-z]{3,}")
SEP = "|"


def toks(t: object) -> list[str]:
    return WORD.findall(str(t).lower())


def primary_scenario(s: object) -> str:
    parts = [p.strip() for p in str(s or "").split(SEP) if p.strip()]
    return parts[0] if parts else "general_unspecified"


# --------------------------------------------------------------------------- #
# Embedding backends: .encode(list[str]) -> np.ndarray (n, d), L2-normalised.
# --------------------------------------------------------------------------- #
class TfidfEmbed:
    name = "tfidf"

    def __init__(self, fit_texts):
        import math
        self.N = len(fit_texts)
        df = defaultdict(int)
        self.docs = [toks(t) for t in fit_texts]
        for d in self.docs:
            for w in set(d):
                df[w] += 1
        self.idf = {w: math.log((self.N + 1) / (n + 1)) + 1 for w, n in df.items()}

    def _vec(self, text):
        import math
        tf = Counter(toks(text))
        v = {w: (1 + math.log(c)) * self.idf.get(w, 0.0) for w, c in tf.items()}
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / nrm for w, x in v.items()}

    def encode(self, texts):
        return [self._vec(t) for t in texts]  # sparse dicts

    @staticmethod
    def cos(a, b):  # sparse cosine (already normalised)
        if len(a) > len(b):
            a, b = b, a
        return sum(x * b.get(w, 0.0) for w, x in a.items())


class BertEmbed:
    name = "bert"

    def __init__(self, fit_texts, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self.np = np
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        import numpy as np
        return np.asarray(self.model.encode(list(texts), batch_size=64,
                          show_progress_bar=True, normalize_embeddings=True))

    @staticmethod
    def cos(a, b):  # dense, both normalised -> dot
        return float((a * b).sum())


def load_reference(text_csv: Path, post_jsonl: Path):
    df = pd.read_csv(text_csv, dtype=str, keep_default_na=False)
    ptext = {}
    for _, r in df.iterrows():
        pid = str(r["post_id"]).strip()
        if pid and pid not in ptext:
            ptext[pid] = str(r.get("post_context", "")).strip()
    lab = {}
    for line in post_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        pid = str(o.get("post_id", "")).strip()
        if pid:
            lab[pid] = primary_scenario(o.get("scenarios", ""))
    ref = [(pid, ptext[pid], lab[pid]) for pid in lab if pid in ptext and ptext[pid]]
    return ref, lab


def load_candidates(cand_jsonl: Path, exclude: set[str]):
    out = []
    for line in cand_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        pid = str(o.get("post_id", "")).strip()
        txt = str(o.get("post_text", "")).strip()
        if pid and txt and pid not in exclude:
            out.append((pid, txt, int(o.get("n_comments", 0))))
    return out


def run(text_csv: Path, post_jsonl: Path, cand_jsonl: Path, backend: str,
        k: int, out_dir: Path) -> None:
    ref, lab = load_reference(text_csv, post_jsonl)
    cands = load_candidates(cand_jsonl, exclude=set(lab))  # exclude already-labelled
    print(f"[ref] {len(ref)} reference posts | [cand] {len(cands)} candidates "
          f"(excluded {sum(1 for _ in lab)} already-labelled overlaps)")

    ref_ids = [r[0] for r in ref]
    ref_lab = [r[2] for r in ref]
    Emb = TfidfEmbed if backend == "tfidf" else BertEmbed
    emb = Emb([r[1] for r in ref])  # fit on reference corpus (tfidf needs idf)
    ref_vecs = emb.encode([r[1] for r in ref])
    cand_vecs = emb.encode([c[1] for c in cands])

    # prototype per scenario
    scen_list = sorted(set(ref_lab))
    if backend == "tfidf":
        protos = {}
        for s in scen_list:
            agg = defaultdict(float)
            cnt = 0
            for v, l in zip(ref_vecs, ref_lab):
                if l == s:
                    for w, x in v.items():
                        agg[w] += x
                    cnt += 1
            if cnt:
                protos[s] = {w: x / cnt for w, x in agg.items()}
        cos = TfidfEmbed.cos
    else:
        import numpy as np
        protos = {}
        for s in scen_list:
            idx = [i for i, l in enumerate(ref_lab) if l == s]
            protos[s] = ref_vecs[idx].mean(axis=0)
        cos = BertEmbed.cos

    rows = []
    proto_bucket = Counter()
    knn_bucket = Counter()
    highconf_bucket = Counter()
    for (pid, txt, ncom), cv in zip(cands, cand_vecs):
        # prototype
        psc = max(scen_list, key=lambda s: cos(cv, protos[s]))
        # knn
        sims = sorted(((cos(cv, rv), ref_lab[i]) for i, rv in enumerate(ref_vecs)),
                      key=lambda x: -x[0])[:k]
        knn_sc = Counter(l for _, l in sims).most_common(1)[0][0]
        conf = "high" if psc == knn_sc else "low"
        rows.append({"post_id": pid, "n_comments": ncom,
                     "scenario_prototype": psc, "scenario_knn": knn_sc,
                     "confidence": conf})
        proto_bucket[psc] += 1
        knn_bucket[knn_sc] += 1
        if conf == "high":
            highconf_bucket[psc] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"candidate_scenarios_{backend}.csv", index=False)

    # current graph scenario distribution (for the balance target)
    cur = Counter(lab.values())
    print(f"\n=== Per-scenario counts | backend={backend} | k={k} ===")
    print(f"{'scenario':28} {'current':>8} {'cand(proto)':>12} {'cand(knn)':>10} {'cand(highconf)':>14}")
    for s in sorted(set(scen_list) | set(proto_bucket), key=lambda x: cur.get(x, 0)):
        print(f"{s:28} {cur.get(s,0):>8} {proto_bucket.get(s,0):>12} "
              f"{knn_bucket.get(s,0):>10} {highconf_bucket.get(s,0):>14}")
    n_high = sum(1 for r in rows if r["confidence"] == "high")
    print(f"\nhigh-confidence (prototype==knn): {n_high}/{len(rows)} "
          f"({n_high/max(len(rows),1):.0%})")
    summary = {"backend": backend, "k": k, "n_ref": len(ref), "n_cand": len(cands),
               "n_highconf": n_high,
               "current": dict(cur), "cand_proto": dict(proto_bucket),
               "cand_knn": dict(knn_bucket), "cand_highconf": dict(highconf_bucket)}
    (out_dir / f"scenario_bucket_summary_{backend}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_dir / f'candidate_scenarios_{backend}.csv'}")
    print(f"saved -> {out_dir / f'scenario_bucket_summary_{backend}.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-csv", type=Path,
                    default=Path("out/annotation_input_merged_v2.csv"))
    ap.add_argument("--post-jsonl", type=Path,
                    default=Path("out/llm_post_problem_annotations_full.jsonl"))
    ap.add_argument("--candidates", type=Path,
                    default=Path("out/deep_hub_candidates_2025_min10.jsonl"))
    ap.add_argument("--backend", default="tfidf", choices=["tfidf", "bert"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=Path("out/scenario_classify"))
    a = ap.parse_args()
    run(a.text_csv, a.post_jsonl, a.candidates, a.backend, a.k, a.out_dir)


if __name__ == "__main__":
    main()
