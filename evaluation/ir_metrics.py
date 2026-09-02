"""Full IR metric set for ranked retrieval — shared by all eval scripts.

A single, dependency-free implementation of the standard information-retrieval
metrics so every evaluation script (two_graph_compare, ablation, retrieval_metrics)
reports the SAME numbers under the SAME names. Binary relevance (a comment is
either gold or not), which matches this corpus (a post's reply comments = gold).

Metrics (all take a ranked list of comment_ids + a gold set):
  Recall@k      : fraction of gold found in the top-k.
  Precision@k   : fraction of the top-k that is gold.
  F1@k          : harmonic mean of P@k and R@k.
  Hit@k         : 1 if >=1 gold in top-k else 0  (a.k.a. Success@k).
  MRR           : 1 / rank of the first gold (early precision).
  nDCG@k        : binary-gain nDCG, log2 discount.
  graded_nDCG@k : graded-gain nDCG for LLM/human utility labels.
  AP (MAP)      : average precision over all gold ranks (MAP = mean of AP).

`eval_full` returns a flat dict with every metric for the default k-set, so a
caller can `accum.append(eval_full(ranked, gold))` then `mean_metrics(rows)`.
"""
from __future__ import annotations

import math
import statistics as st


def recall_at(ranked, gold, k):
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def precision_at(ranked, gold, k):
    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & gold) / k


def f1_at(ranked, gold, k):
    p = precision_at(ranked, gold, k)
    r = recall_at(ranked, gold, k)
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def hit_at(ranked, gold, k):
    """1.0 if at least one gold in top-k (Success@k), else 0.0."""
    return 1.0 if set(ranked[:k]) & gold else 0.0


def mrr(ranked, gold):
    for i, c in enumerate(ranked, 1):
        if c in gold:
            return 1.0 / i
    return 0.0


def ndcg_at(ranked, gold, k):
    dcg = sum(1.0 / math.log2(i + 2) for i, c in enumerate(ranked[:k]) if c in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0


def graded_ndcg_at(ranked, gains, k):
    """nDCG@k for a mapping ``item_id -> non-negative graded relevance``.

    Uses the conventional exponential gain ``2**rel - 1`` and the same log2
    rank discount as :func:`ndcg_at`.  Keeping this in the canonical metrics
    module prevents LLM/human utility evaluations from drifting across scripts.
    """
    gain_values = [max(0.0, float(gains.get(item, 0.0))) for item in ranked[:k]]
    ideal_values = sorted((max(0.0, float(v)) for v in gains.values()), reverse=True)[:k]
    dcg = sum((2.0 ** gain - 1.0) / math.log2(i + 2)
              for i, gain in enumerate(gain_values))
    ideal = sum((2.0 ** gain - 1.0) / math.log2(i + 2)
                for i, gain in enumerate(ideal_values))
    return dcg / ideal if ideal > 0 else 0.0


def average_precision(ranked, gold):
    """AP over all gold items (0 for unretrieved gold). Mean over queries = MAP."""
    if not gold:
        return 0.0
    hits = 0
    cum = 0.0
    for i, c in enumerate(ranked, 1):
        if c in gold:
            hits += 1
            cum += hits / i
    return cum / len(gold)


# default k-sets — cover early precision (1/5/10) and deep recall (100)
DEFAULT_RECALL_KS = (10, 100)
DEFAULT_PREC_KS = (10,)
DEFAULT_HIT_KS = (1, 5, 10)


def eval_full(ranked, gold, recall_ks=DEFAULT_RECALL_KS,
              prec_ks=DEFAULT_PREC_KS, hit_ks=DEFAULT_HIT_KS):
    """Flat metric dict for one ranked list. Keys are paper-ready column names."""
    gold = set(gold)
    res = {
        "MRR": mrr(ranked, gold),
        "MAP": average_precision(ranked, gold),
        "nDCG@10": ndcg_at(ranked, gold, 10),
        "nDCG@100": ndcg_at(ranked, gold, 100),
    }
    for k in recall_ks:
        res[f"Recall@{k}"] = recall_at(ranked, gold, k)
    for k in prec_ks:
        res[f"P@{k}"] = precision_at(ranked, gold, k)
        res[f"F1@{k}"] = f1_at(ranked, gold, k)
    for k in hit_ks:
        res[f"Hit@{k}"] = hit_at(ranked, gold, k)
    return res


def mean_metrics(rows, ndigits=4):
    """Mean over a list of eval_full dicts. Preserves key order of the first row."""
    if not rows:
        return {}
    keys = list(rows[0].keys())
    return {k: round(st.mean(r[k] for r in rows), ndigits) for k in keys}


# --------------------------------------------------------------------------- #
# Backward-compatible shim: same keys the old backends.eval_ranking returned
# (rr / ndcg10 / recall{k}). Lets existing callers (ablation.py) switch to the
# single source without changing their key access. New code: use eval_full.
# --------------------------------------------------------------------------- #
def eval_ranking(ranked, gold, Ks=(10, 100)):
    gold = set(gold)
    res = {"rr": mrr(ranked, gold), "ndcg10": ndcg_at(ranked, gold, 10)}
    for k in Ks:
        res[f"recall{k}"] = recall_at(ranked, gold, k)
    return res
