"""Score one or more retrieval outputs (HippoRAG or dense baseline) against
MuSiQue gold supporting titles, using the canonical evaluation.ir_metrics.

Positive-control read-out: graph (HippoRAG) should beat dense on multi-hop
Recall@k / nDCG. A paired Recall@5 delta + bootstrap CI is reported per system
pair.  This tests HARNESS SENSITIVITY, not the bespoke ADHD graph method.

Run:
  python -m evaluation.score_multihop_retrieval \
    --gold out/positive_control/musique_150/gold.json \
    --pred hipporag=out/positive_control/musique_150/hipporag/retrieval.jsonl \
    --pred dense=out/positive_control/musique_150/dense_tfidf.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _load_pred(path):
    """id -> retrieved_titles (accepts jsonl or json list; tolerant field names)."""
    text = Path(path).read_text(encoding="utf-8").strip()
    rows = (json.loads(text) if text.startswith("[")
            else [json.loads(l) for l in text.splitlines() if l.strip()])
    out = {}
    for r in rows:
        rid = str(r.get("id") or r.get("query_id") or r.get("qid"))
        titles = r.get("retrieved_titles") or r.get("titles") or r.get("retrieved") or []
        out[rid] = [str(t) for t in titles]
    return out


def score_system(preds, gold):
    """Per-query eval_full rows + mean, over queries present in gold."""
    from evaluation.ir_metrics import eval_full, mean_metrics
    rows, per_q_recall5 = [], {}
    joint = {2: [], 5: [], 10: [], 20: []}
    for qid, g in gold.items():
        if qid not in preds:
            continue
        gset = set(g)
        if not gset:
            continue
        ranked = preds[qid]
        rows.append(eval_full(ranked, gset, recall_ks=(2, 5, 10, 20)))
        per_q_recall5[qid] = rows[-1].get("Recall@5", 0.0)
        for k in joint:
            joint[k].append(float(gset.issubset(set(ranked[:k]))))
    summary = mean_metrics(rows)
    summary.update({f"Joint@{k}": round(sum(values) / len(values), 4)
                    for k, values in joint.items() if values})
    return summary, per_q_recall5, len(rows)


def paired_bootstrap_delta(a_recall, b_recall, n_boot=2000, seed=20260711):
    """Paired bootstrap 95% CI on mean(a-b) over shared queries (Recall@5)."""
    ids = [q for q in a_recall if q in b_recall]
    diffs = [a_recall[q] - b_recall[q] for q in ids]
    if not diffs:
        return None
    rng = random.Random(seed)
    means = []
    n = len(diffs)
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return {"delta_mean": round(sum(diffs) / n, 4),
            "ci95": [round(lo, 4), round(hi, 4)], "n": n}


def per_query_joint(preds, gold, k):
    """Return query-level all-supporting-evidence success for paired tests."""
    out = {}
    for qid, g in gold.items():
        if qid not in preds or not g:
            continue
        out[qid] = float(set(g).issubset(set(preds[qid][:k])))
    return out


def per_query_ir_metric(preds, gold, metric, recall_ks=(5, 10, 20)):
    """Canonical query-level IR values for arbitrary paired comparisons."""
    from evaluation.ir_metrics import eval_full
    out = {}
    for qid, g in gold.items():
        if qid not in preds or not g:
            continue
        row = eval_full(preds[qid], set(g), recall_ks=recall_ks)
        if metric not in row:
            raise KeyError(f"metric not emitted by canonical eval_full: {metric}")
        out[qid] = float(row[metric])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", action="append", required=True,
                    help="name=path (repeatable), e.g. hipporag=.../retrieval.jsonl")
    ap.add_argument("--out", type=Path,
                    help="optional JSON manifest path for the complete comparison")
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    systems = {}
    recalls = {}
    for spec in args.pred:
        name, _, path = spec.partition("=")
        preds = _load_pred(path)
        summary, per_q, n = score_system(preds, gold)
        systems[name] = {"summary": summary, "n_scored": n}
        recalls[name] = per_q

    pairwise = []
    result = {"systems": systems}

    names = list(systems)
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            d = paired_bootstrap_delta(recalls[names[i]], recalls[names[j]])
            if d:
                pairwise.append({"left": names[i], "right": names[j],
                                 "metric": "Recall@5", **d})
                print(f"[Recall@5 paired] {names[i]} - {names[j]}: "
                      f"{d['delta_mean']:+.4f}  CI95 {d['ci95']}  (n={d['n']})")
    result["paired"] = pairwise
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
