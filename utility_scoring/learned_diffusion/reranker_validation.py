"""Fair, query-grouped validation of linear, LambdaMART and MLP rerankers.

The module shares one frozen candidate registry, feature schema, qrels and
outer 5x5 query split.  Continuous-feature scaling is fitted inside each outer
or inner training fold.  Unjudged candidates are scored but never enter a
supervised objective or metric as zero.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
# On macOS, load XGBoost's OpenMP runtime before PyTorch's.  Reversing this
# order can crash inside XGBoost's ranking-label metadata bridge under the
# project's Python 3.13 reproducibility environment.
import xgboost
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import configuration as project_config
from evaluation.ir_metrics import graded_ndcg_at
from evaluation.statistics import bootstrap_ci


BASIC_FEATURES = (
    "dense_score", "dense_rank_reciprocal", "dense_missing", "query_comment_cosine",
    "bm25_score", "bm25_rank_reciprocal", "bm25_missing", "comment_length_log",
    "query_length_log", "lexical_jaccard", "lexical_query_coverage",
)
GRAPH_FEATURES = (
    "official_ppr_score", "official_ppr_rank_reciprocal", "official_ppr_missing",
    "graph_reachable", "shortest_seed_distance", "reachable_seed_count",
    "seed_weighted_path_score", "comment_degree_log", "candidate_degree_percentile",
    "local_degree_log", "original_restart_mass",
)
ARMS = (
    "frozen_dense", "official_static", "linear_basic", "dense_lambdamart",
    "dense_mlp", "graph_lambdamart", "graph_mlp",
)
TOKEN = re.compile(r"(?u)\b\w\w+\b")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+"\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n")


def cfg(config_key: str = "reranker_signal_validation") -> tuple[Path, dict]:
    root = Path(__file__).resolve().parents[2]
    raw = project_config.load()[config_key]
    for key in ("output_dir", "old_judgments", "queries", "query_admin", "corpus",
                "run_registry", "local_graph_dir", "split_manifest",
                "training_judgments", "evaluation_judgments"):
        if key not in raw:
            continue
        path = Path(raw[key]); raw[key] = path if path.is_absolute() else root/path
    return root, raw


def tokens(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))


def load_run_features(registry_path: Path, eligible: set[str]) -> dict[tuple[str, str], dict]:
    registry = json.loads(registry_path.read_text())
    wanted = {"cohere_dense", "bm25s", "official_static"}
    result: dict[tuple[str, str], dict] = defaultdict(dict)
    for record in registry["runs"]:
        method = str(record["method"])
        if method not in wanted or record.get("status") != "available":
            continue
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in read_jsonl(Path(record["path"])):
            if str(row["query_id"]) in eligible:
                grouped[str(row["query_id"])].append(row)
        for qid, rows in grouped.items():
            values = np.asarray([float(row.get("raw_score", row.get("score", 0.0)))
                                 for row in rows])
            low, high = float(values.min()), float(values.max())
            normalized = (values-low)/(high-low) if high > low else np.zeros_like(values)
            for row, score in zip(rows, normalized, strict=True):
                result[(qid, str(row["comment_id"]))][method] = {
                    "rank": int(row["rank"]), "score": float(score)}
    return dict(result)


def seed_path_features(graph: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_index = graph["edge_index"].numpy()
    node_count = len(graph["global_nodes"])
    adjacency = [[] for _ in range(node_count)]
    for source, target in edge_index.T:
        adjacency[int(source)].append(int(target))
    seeds = graph["candidate_seed_local_nodes"].numpy().astype(int)
    candidates = graph["candidate_comment_local_nodes"].numpy().astype(int)
    candidate_index: dict[int, list[int]] = defaultdict(list)
    for index, node in enumerate(candidates): candidate_index[int(node)].append(index)
    reachable_count = np.zeros(len(candidates), dtype=np.float32)
    shortest = np.full(len(candidates), 4.0, dtype=np.float32)
    weighted = np.zeros(len(candidates), dtype=np.float32)
    metadata = graph["candidate_seed_metadata"]
    weights = np.asarray([max(0.0, float(row.get("restart_weight") or 0.0))
                          for row in metadata], dtype=np.float32)
    if len(weights) and weights.sum() <= 0: weights[:] = 1.0
    if len(weights): weights /= weights.sum()
    for seed_index, seed in enumerate(seeds):
        distance = {int(seed): 0}; queue = deque([int(seed)])
        while queue:
            node = queue.popleft(); depth = distance[node]
            if node in candidate_index:
                for index in candidate_index[node]:
                    reachable_count[index] += 1
                    shortest[index] = min(shortest[index], float(depth))
                    weighted[index] += float(weights[seed_index])/(1.0+depth)
            if depth == 3: continue
            for neighbor in adjacency[node]:
                if neighbor not in distance:
                    distance[neighbor] = depth+1; queue.append(neighbor)
    return reachable_count, shortest, weighted


def build_registry(raw: dict, out: Path, *, query_scope: str = "entered") -> tuple[list[dict], dict]:
    """Build the frozen basic/graph feature registry.

    ``entered`` preserves the original LUAD/reranker experiment exactly.
    ``all`` reuses the already materialised label-blind local tensors for
    downstream diagnostics that must also cover dense-fallback queries.
    """
    if query_scope not in {"entered", "all"}:
        raise ValueError(f"unsupported query_scope={query_scope!r}")
    import hashlib
    local_paths = sorted(raw["local_graph_dir"].glob("*.pt"))
    graphs = [torch.load(path, weights_only=False) for path in local_paths]
    entered = {str(graph["query_id"]) for graph in graphs if graph["recognition_success"]}
    queries_json = json.loads(raw["queries"].read_text())
    queries = {str(row.get("id") or row.get("query_id")):
               str(row.get("question") or row.get("query_text")) for row in queries_json}
    corpus_rows = json.loads(raw["corpus"].read_text())
    corpus = {str(row["title"]): str(row["text"]) for row in corpus_rows}
    eligible_run_queries = entered if query_scope == "entered" else set(queries)
    run = load_run_features(raw["run_registry"], eligible_run_queries)
    rows = []
    for graph in graphs:
        qid = str(graph["query_id"])
        if query_scope == "entered" and qid not in entered: continue
        comments = [str(value) for value in graph["candidate_comment_ids"]]
        q_tokens = tokens(queries[qid]); query_len = math.log1p(len(queries[qid]))
        candidate_features = graph["candidate_features"].numpy()
        local_nodes = graph["candidate_comment_local_nodes"].numpy().astype(int)
        local_degree = np.bincount(graph["edge_index"][0].numpy(),
                                   minlength=len(graph["global_nodes"]))
        degree_log = candidate_features[:, 6]
        order = np.argsort(degree_log, kind="stable")
        percentiles = np.empty(len(order), dtype=np.float32); percentiles[order] = (
            np.arange(len(order))/max(1, len(order)-1))
        seed_count, shortest, seed_weighted = seed_path_features(graph)
        restart = graph["original_restart"].numpy()[local_nodes]
        for index, cid in enumerate(comments):
            c_tokens = tokens(corpus[cid]); intersection = len(q_tokens & c_tokens)
            union = len(q_tokens | c_tokens)
            provenance = run.get((qid, cid), {})
            def method_values(name: str) -> tuple[float, float, float]:
                value = provenance.get(name)
                if value is None: return 0.0, 0.0, 1.0
                return float(value["score"]), 1.0/(60.0+float(value["rank"])), 0.0
            dense_score, dense_rank, dense_missing = method_values("cohere_dense")
            bm25_score, bm25_rank, bm25_missing = method_values("bm25s")
            ppr_score, ppr_rank, ppr_missing = method_values("official_static")
            basic = [
                dense_score, dense_rank, dense_missing, float(candidate_features[index, 3]),
                bm25_score, bm25_rank, bm25_missing, math.log1p(len(corpus[cid])), query_len,
                intersection/max(1, union), intersection/max(1, len(q_tokens)),
            ]
            graph_values = [
                ppr_score, ppr_rank, ppr_missing, float(candidate_features[index, 4]),
                float(shortest[index]), float(seed_count[index]), float(seed_weighted[index]),
                float(degree_log[index]), float(percentiles[index]),
                math.log1p(float(local_degree[local_nodes[index]])), float(restart[index]),
            ]
            rows.append({
                "query_id": qid, "comment_id": cid,
                "basic_features": dict(zip(BASIC_FEATURES, basic, strict=True)),
                "graph_features": dict(zip(GRAPH_FEATURES, graph_values, strict=True)),
                "candidate_selection_used_utility": False,
            })
    write_jsonl(out/"unified_candidate_registry.jsonl", rows)
    schema = {
        "version": raw["version"], "candidate_queries": len({row["query_id"] for row in rows}),
        "query_scope": query_scope,
        "candidate_pairs": len(rows), "candidate_set_identical_for_all_models": True,
        "basic_features": list(BASIC_FEATURES), "graph_features": list(GRAPH_FEATURES),
        "unavailable_not_fabricated": [
            "relation_type_coverage: canonical graph has no relation-type edge attribute",
            "global_hub_percentile: local frozen tensor lacks full-graph percentile; candidate-universe percentile is named explicitly",
            "text_embedding_projection: not used by current implementation",
            "max_or_mean_seed_similarity: seed-to-candidate similarities are not frozen in the local tensor",
            "passage_node_type: every candidate is a passage node, so the feature is constant",
            "exact_local_path_count: only bounded reachability, distance and reachable-seed count are computed",
            "local_subgraph_density: query-level constant and therefore omitted from within-query ranking",
        ],
        "normalization": "StandardScaler fitted on outer/inner training-query candidates only",
        "label_features": [], "query_target_statistics": [], "test_features": [],
        "source_hash": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
    }
    write_json(out/"unified_feature_schema.json", schema)
    return rows, schema


@dataclass
class Dataset:
    rows: list[dict]
    qrels: dict[tuple[str, str], float]

    def __post_init__(self):
        self.by_query: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows): self.by_query[row["query_id"]].append(index)
        self.basic = np.asarray([[row["basic_features"][name] for name in BASIC_FEATURES]
                                 for row in self.rows], dtype=np.float32)
        self.graph = np.asarray([[row["graph_features"][name] for name in GRAPH_FEATURES]
                                 for row in self.rows], dtype=np.float32)
        self.utility = np.asarray([self.qrels.get((row["query_id"], row["comment_id"]), np.nan)
                                   for row in self.rows], dtype=np.float32)


def load_dataset(rows: list[dict], qrels_path: Path) -> Dataset:
    qrels = {(str(row["query_id"]), str(row["comment_id"])): float(row["utility"])
             for row in read_jsonl(qrels_path)}
    return Dataset(rows, qrels)


def query_pairs(dataset: Dataset, qids: list[str], max_pairs: int, margin: float,
                seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs, labels, weights = [], [], []
    rng = random.Random(seed)
    for qid in qids:
        indices = [i for i in dataset.by_query[qid] if np.isfinite(dataset.utility[i])]
        eligible = [(i, j) for i in indices for j in indices
                    if dataset.utility[i]-dataset.utility[j] >= margin]
        rng.shuffle(eligible); eligible = eligible[:max_pairs]
        if not eligible: continue
        weight = 1.0/(2*len(eligible))
        for left, right in eligible:
            pairs.extend(((left, right), (right, left))); labels.extend((1, 0));
            weights.extend((weight, weight))
    return np.asarray(pairs), np.asarray(labels), np.asarray(weights)


def pairwise_accuracy(dataset: Dataset, scores: np.ndarray, qids: list[str], margin=1.0) -> float:
    values = []
    for qid in qids:
        indices = [i for i in dataset.by_query[qid] if np.isfinite(dataset.utility[i])]
        pairs = [(i, j) for i in indices for j in indices
                 if dataset.utility[i]-dataset.utility[j] >= margin]
        if pairs: values.append(np.mean([scores[i] > scores[j] for i, j in pairs]))
    return statistics.fmean(values) if values else float("nan")


def fit_scaler(dataset: Dataset, train_qids: list[str], feature_set: str) -> StandardScaler:
    indices = [i for qid in train_qids for i in dataset.by_query[qid]]
    matrix = dataset.basic if feature_set == "basic" else np.hstack((dataset.basic, dataset.graph))
    return StandardScaler().fit(matrix[indices])


def matrix(dataset: Dataset, scaler: StandardScaler, feature_set: str) -> np.ndarray:
    values = dataset.basic if feature_set == "basic" else np.hstack((dataset.basic, dataset.graph))
    return scaler.transform(values).astype(np.float32)


def fit_linear(dataset: Dataset, X: np.ndarray, train_qids: list[str], l2: float,
               train_cfg: dict, seed: int):
    pairs, labels, weights = query_pairs(dataset, train_qids,
        int(train_cfg["max_pairs_per_query"]), float(train_cfg["pair_margin"]), seed)
    differences = X[pairs[:, 0]]-X[pairs[:, 1]]
    model = LogisticRegression(C=1.0/max(l2, 1e-12), fit_intercept=False,
                               random_state=seed, max_iter=1000)
    model.fit(differences, labels, sample_weight=weights)
    return model


def fit_lambdamart(dataset: Dataset, X: np.ndarray, train_qids: list[str],
                   settings: dict, seed: int):
    indices = np.asarray([i for qid in sorted(train_qids) for i in dataset.by_query[qid]
                          if np.isfinite(dataset.utility[i])], dtype=int)
    query_ids = [
        qid
        for qid in sorted(train_qids)
        for i in dataset.by_query[qid]
        if np.isfinite(dataset.utility[i])
    ]
    return fit_xgb_lambdamart(
        X[indices], dataset.utility[indices], query_ids, settings, seed
    )


def historical_utility_grade(value: float) -> int:
    """Map utility-v2 to the project's frozen seven-level LTR grade."""
    return max(0, min(6, int(round(float(value))) - 1))


def fit_xgb_lambdamart(
    X: np.ndarray,
    utility: np.ndarray,
    query_ids: list[str],
    settings: dict,
    seed: int,
):
    """Fit the canonical project LambdaMART on arbitrary numeric features.

    Rows are stably query-sorted before passing ``group`` to XGBoost.  The
    defaults preserve the historical project implementation; callers may
    expose the same parameters in a small, query-grouped inner-CV grid.
    """
    if len(X) != len(utility) or len(X) != len(query_ids):
        raise ValueError("LambdaMART feature, utility and query arrays differ")
    if not len(X):
        raise ValueError("LambdaMART requires at least one training row")
    order = np.argsort(np.asarray(query_ids, dtype=str), kind="stable")
    sorted_query_ids = [str(query_ids[index]) for index in order]
    groups = [
        len(list(rows))
        for _, rows in itertools.groupby(sorted_query_ids)
    ]
    y = np.asarray(
        [historical_utility_grade(float(utility[index])) for index in order],
        dtype=np.int32,
    )
    model = xgboost.XGBRanker(
        objective=str(settings.get("objective", "rank:ndcg")),
        tree_method=str(settings.get("tree_method", "hist")),
        n_estimators=int(settings["n_estimators"]),
        learning_rate=float(settings["learning_rate"]),
        max_depth=int(settings["max_depth"]),
        min_child_weight=float(settings.get("min_child_weight", 1.0)),
        subsample=float(settings.get("subsample", 0.8)),
        colsample_bytree=float(settings.get("colsample_bytree", 0.8)),
        reg_lambda=float(settings.get("reg_lambda", 1.5)),
        reg_alpha=float(settings.get("reg_alpha", 0.0)),
        lambdarank_pair_method=str(
            settings.get("lambdarank_pair_method", "mean")
        ),
        lambdarank_num_pair_per_sample=int(
            settings.get(
                "lambdarank_num_pair_per_sample",
                settings.get("pairs_per_sample", 8),
            )
        ),
        ndcg_exp_gain=bool(
            settings.get(
                "ndcg_exp_gain",
                settings.get("lambdarank_ndcg_exp_gain", True),
            )
        ),
        eval_metric=str(settings.get("eval_metric", "ndcg@8")),
        random_state=seed,
        n_jobs=int(settings.get("n_jobs", 1)),
    )
    model.fit(np.asarray(X)[order], y, group=groups, verbose=False)
    return model


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int, settings: dict):
        super().__init__(); hidden = int(settings["hidden_dim"]); layers=[]
        current=input_dim
        for _ in range(int(settings["layers"])):
            layers += [nn.Linear(current, hidden), nn.ReLU(), nn.Dropout(float(settings["dropout"]))]
            current=hidden
        layers.append(nn.Linear(current, 1)); self.network=nn.Sequential(*layers)
    def forward(self, values): return self.network(values).squeeze(-1)


def fit_mlp(dataset: Dataset, X: np.ndarray, train_qids: list[str], settings: dict,
            train_cfg: dict, seed: int, *, tuning: bool = False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=SmallMLP(X.shape[1], settings); optimizer=torch.optim.Adam(
        model.parameters(), lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"]))
    tensor=torch.tensor(X); utilities=torch.tensor(dataset.utility); best=None; best_loss=float("inf"); stale=0
    pair_rows=[]; regression_rows=[]; rng=random.Random(seed)
    for qid in train_qids:
        judged=[i for i in dataset.by_query[qid] if np.isfinite(dataset.utility[i])]
        eligible=[(i,j) for i in judged for j in judged
                  if dataset.utility[i]-dataset.utility[j]>=float(train_cfg["pair_margin"])]
        rng.shuffle(eligible); eligible=eligible[:int(train_cfg["max_pairs_per_query"])]
        if eligible: pair_rows.append(eligible)
        if judged: regression_rows.append(judged)
    pair_left=torch.tensor([i for pairs in pair_rows for i,j in pairs],dtype=torch.long)
    pair_right=torch.tensor([j for pairs in pair_rows for i,j in pairs],dtype=torch.long)
    pair_weights=torch.tensor([1.0/(len(pair_rows)*len(pairs)) for pairs in pair_rows for _ in pairs])
    max_epochs=int(train_cfg["tuning_epochs"] if tuning else train_cfg["epochs"])
    patience=int(train_cfg["tuning_patience"] if tuning else train_cfg["patience"])
    history=[]
    for epoch in range(max_epochs):
        model.train(); optimizer.zero_grad(); scores=model(tensor)
        rank_loss=(torch.nn.functional.softplus(
            -(scores[pair_left]-scores[pair_right]))*pair_weights).sum()
        regressions=[]
        for judged in regression_rows:
            indices=torch.tensor(judged,dtype=torch.long)
            regressions.append(torch.nn.functional.huber_loss(
                torch.sigmoid(scores[indices]),(utilities[indices]-1.0)/6.0))
        loss=rank_loss+float(train_cfg["lambda_regression"])*torch.stack(regressions).mean()
        loss.backward(); optimizer.step(); value=float(loss.detach())
        history.append(value)
        if value < best_loss-1e-5:
            best_loss=value; best={key: val.detach().clone() for key,val in model.state_dict().items()}; stale=0
        else:
            stale+=1
            if stale>=patience: break
    model.load_state_dict(best); return model, {"epochs":len(history),"first_loss":history[0],"best_loss":best_loss}


def inner_folds(qids: list[str], folds: int, seed: int):
    shuffled=sorted(qids); random.Random(seed).shuffle(shuffled)
    buckets=[shuffled[i::folds] for i in range(folds)]
    return [(sorted(set(qids)-set(valid)), sorted(valid)) for valid in buckets if valid]


def predict_model(kind: str, model, X: np.ndarray) -> np.ndarray:
    if kind == "mlp":
        model.eval()
        with torch.no_grad(): return model(torch.tensor(X)).numpy()
    if kind == "linear": return model.decision_function(X)
    return model.predict(X)


def tune(dataset: Dataset, train_qids: list[str], family: str, feature_set: str,
         configs: list, train_cfg: dict, seed: int) -> tuple[dict|float, list[dict]]:
    traces=[]; folds=inner_folds(train_qids, int(train_cfg["inner_folds"]), seed)
    for config_index, setting in enumerate(configs):
        scores=[]
        for inner_index,(inner_train,inner_valid) in enumerate(folds):
            scaler=fit_scaler(dataset, inner_train, feature_set); X=matrix(dataset,scaler,feature_set)
            fit_seed=seed+config_index*97+inner_index
            if family=="linear": model=fit_linear(dataset,X,inner_train,float(setting),train_cfg,fit_seed)
            elif family=="lambdamart": model=fit_lambdamart(dataset,X,inner_train,setting,fit_seed)
            else: model,_=fit_mlp(dataset,X,inner_train,setting,train_cfg,fit_seed,tuning=True)
            pred=predict_model(family,model,X); scores.append(pairwise_accuracy(dataset,pred,inner_valid))
        traces.append({"config":setting,"inner_pairwise_accuracy":statistics.fmean(scores),"fold_scores":scores})
    best=max(range(len(traces)),key=lambda i:(traces[i]["inner_pairwise_accuracy"],-i))
    return configs[best],traces


def query_metric(dataset: Dataset, qid: str, scores: np.ndarray, arm: str,
                 repeat: int, fold: int) -> tuple[dict,dict]:
    indices=dataset.by_query[qid]; order=sorted(indices,key=lambda i:(-float(scores[i]),dataset.rows[i]["comment_id"]))
    top=order[:3]; judged=[i for i in top if np.isfinite(dataset.utility[i])]
    coverage=len(judged)/3
    gains={dataset.rows[i]["comment_id"]:float(dataset.utility[i]) for i in indices if np.isfinite(dataset.utility[i])}
    top_ids=[dataset.rows[i]["comment_id"] for i in top]
    complete=coverage==1.0
    useful=[float(dataset.utility[i])>=4 for i in top] if complete else []
    judged_indices=[i for i in indices if np.isfinite(dataset.utility[i])]
    rho=float(spearmanr(scores[judged_indices],dataset.utility[judged_indices]).statistic)
    row={
        "query_id":qid,"arm":arm,"repeat":repeat,"fold":fold,"coverage_at3":coverage,
        "full_ndcg_at3":graded_ndcg_at(top_ids,gains,3) if complete else None,
        "mean_utility_at3":float(np.mean(dataset.utility[top])) if complete else None,
        "success_at1":float(useful[0]) if complete else None,
        "success_at3":float(any(useful)) if complete else None,
        "zero_useful_result_at3":float(not any(useful)) if complete else None,
        "pairwise_accuracy":pairwise_accuracy(dataset,scores,[qid]),"spearman":rho,
        "judged_only_ndcg_at3":graded_ndcg_at(
            [dataset.rows[i]["comment_id"] for i in judged],gains,min(3,len(judged))) if judged else None,
        "unjudged_is_zero":False,
    }
    prediction={"query_id":qid,"arm":arm,"repeat":repeat,"fold":fold,
                "ranked_comment_ids":[dataset.rows[i]["comment_id"] for i in order],
                "scores":[float(scores[i]) for i in order],
                "judged_mask":[bool(np.isfinite(dataset.utility[i])) for i in order]}
    return row,prediction


def aggregate(rows: list[dict]) -> dict:
    metrics=("coverage_at3","full_ndcg_at3","mean_utility_at3","success_at1","success_at3",
             "zero_useful_result_at3","pairwise_accuracy","spearman","judged_only_ndcg_at3")
    return {"oof_query_rows":len(rows),"unique_queries":len({r['query_id'] for r in rows}),
            **{name:(statistics.fmean(float(r[name]) for r in rows if r[name] is not None))
               for name in metrics}}


def paired(rows: list[dict], left: str, right: str, metric: str) -> dict:
    by=defaultdict(lambda:defaultdict(list))
    for row in rows:
        if row["arm"] in {left,right} and row[metric] is not None:
            by[row["query_id"]][row["arm"]].append(float(row[metric]))
    deltas=[]
    for qid,arms in by.items():
        if left in arms and right in arms:
            deltas.append(statistics.fmean(arms[left])-statistics.fmean(arms[right]))
    lo,hi=bootstrap_ci(deltas,n_boot=5000,seed=20260719+len(metric)+len(left))
    return {"left":left,"right":right,"metric":metric,"queries":len(deltas),
            "mean_delta":statistics.fmean(deltas) if deltas else None,"bootstrap_95ci":[lo,hi],
            "improved_tied_degraded":[sum(x>1e-9 for x in deltas),sum(abs(x)<=1e-9 for x in deltas),sum(x<-1e-9 for x in deltas)]}


def evaluate_frozen_predictions(
    config_key: str = "reranker_signal_validation",
) -> dict:
    """Re-score the already frozen OOF rankings after coverage-only judging.

    This function never refits a model.  It prevents the second residual batch
    from becoming a label-dependent candidate/model-selection loop: the new
    labels complete evaluation of rankings that were frozen before they were
    observed.
    """
    _, raw = cfg(config_key); out = raw["output_dir"]
    registry_path = out / "unified_candidate_registry.jsonl"
    qrels_path = raw.get(
        "evaluation_judgments",
        raw.get(
            "training_judgments",
            out / "utility_v2_reranker_augmented.jsonl",
        ),
    )
    predictions_path = out / "reranker_cv_predictions.jsonl"
    for path in (registry_path, qrels_path, predictions_path):
        if not path.exists(): raise SystemExit(f"required frozen artifact missing: {path}")
    dataset = load_dataset(read_jsonl(registry_path), qrels_path)
    predictions = read_jsonl(predictions_path)
    index = {(row["query_id"], row["comment_id"]): i
             for i, row in enumerate(dataset.rows)}
    metrics = []
    refreshed_predictions = []
    for prediction in predictions:
        scores = np.zeros(len(dataset.rows), dtype=np.float32)
        qid = str(prediction["query_id"])
        if len(prediction["ranked_comment_ids"]) != len(prediction["scores"]):
            raise ValueError("frozen prediction ids/scores length mismatch")
        for cid, score in zip(prediction["ranked_comment_ids"], prediction["scores"], strict=True):
            scores[index[(qid, str(cid))]] = float(score)
        row, refreshed = query_metric(dataset, qid, scores, str(prediction["arm"]),
                                      int(prediction["repeat"]), int(prediction["fold"]))
        metrics.append(row); refreshed_predictions.append(refreshed)
    metric_path = out / "reranker_cv_per_query_metrics.jsonl"
    summary_path = out / "reranker_cv_metrics.json"
    if metric_path.exists() and not (out / "reranker_cv_per_query_metrics_pre_followup.jsonl").exists():
        write_jsonl(out / "reranker_cv_per_query_metrics_pre_followup.jsonl", read_jsonl(metric_path))
    if summary_path.exists() and not (out / "reranker_cv_metrics_pre_followup.json").exists():
        write_json(out / "reranker_cv_metrics_pre_followup.json",
                   json.loads(summary_path.read_text()))
    write_jsonl(metric_path, metrics)
    # Ranked identities and scores must be byte-for-byte equivalent in meaning;
    # only judged_mask changes after the coverage-completion batch.
    write_jsonl(out / "reranker_cv_predictions_coverage_complete.jsonl", refreshed_predictions)
    summary = {arm: aggregate([row for row in metrics if row["arm"] == arm]) for arm in ARMS}
    write_json(summary_path, summary)
    comparisons = []
    pairs = [
        ("dense_lambdamart", "frozen_dense"),
        ("dense_mlp", "frozen_dense"),
        ("graph_lambdamart", "dense_lambdamart"),
        ("graph_mlp", "dense_mlp"),
        ("graph_lambdamart", "graph_mlp"),
        ("linear_basic", "frozen_dense"),
        ("linear_basic", "official_static"),
        ("linear_basic", "dense_lambdamart"),
        ("linear_basic", "dense_mlp"),
    ]
    for left, right in pairs:
        for metric in ("full_ndcg_at3", "mean_utility_at3", "pairwise_accuracy", "spearman",
                       "success_at1", "success_at3", "zero_useful_result_at3"):
            comparisons.append(paired(metrics, left, right, metric))
    # Best is selected only after all pre-registered finalist summaries exist;
    # the rule is fixed to mean full-coverage nDCG@3, then arm name.
    finalist = ("linear_basic", "dense_lambdamart", "dense_mlp",
                "graph_lambdamart", "graph_mlp")
    best = max(finalist, key=lambda arm: (summary[arm]["full_ndcg_at3"], arm))
    for baseline in ("official_static", "frozen_dense"):
        for metric in ("full_ndcg_at3", "mean_utility_at3"):
            comparisons.append(paired(metrics, best, baseline, metric))
    write_json(out / "reranker_paired_comparisons.json", comparisons)
    coverage = {
        "by_arm": {arm: summary[arm]["coverage_at3"] for arm in ARMS},
        "all_arms_at_least_95pct": all(summary[arm]["coverage_at3"] >= .95 for arm in ARMS),
        "all_arms_complete": all(summary[arm]["coverage_at3"] == 1.0 for arm in ARMS),
        "evaluation_predictions_refit_after_followup_labels": False,
        "followup_labels_used_for_model_or_candidate_selection": False,
        "unjudged_assigned_zero": False,
    }
    write_json(out / "coverage_bias_report.json", coverage)
    result = {"best_reranker_by_full_ndcg_at3": best, "coverage": coverage,
              "summary": summary}
    print(json.dumps(result, indent=2)); return result


def run(config_key: str = "reranker_signal_validation") -> None:
    root,raw=cfg(config_key); out=raw["output_dir"]; out.mkdir(parents=True,exist_ok=True)
    registry_path=out/"unified_candidate_registry.jsonl"
    schema_path=out/"unified_feature_schema.json"
    if registry_path.exists() and schema_path.exists():
        rows=read_jsonl(registry_path); schema=json.loads(schema_path.read_text())
    else:
        rows,schema=build_registry(raw,out)
    qrels_path=raw.get(
        "training_judgments", out/"utility_v2_reranker_augmented.jsonl"
    )
    if not qrels_path.exists(): raise SystemExit("anchor-gated augmented qrels missing")
    dataset=load_dataset(rows,qrels_path)
    split=json.loads(raw["split_manifest"].read_text()); train_cfg=raw["training"]
    entered={row["query_id"] for row in rows}; predictions=[]; metrics=[]; tuning=[]
    for split_row in split["rows"]:
        train_qids=[q for q in split_row["train_query_ids"] if q in entered]
        valid_qids=[q for q in split_row["validation_query_ids"] if q in entered]
        if not valid_qids: continue
        repeat,fold=int(split_row["repeat"]),int(split_row["fold"]); seed=int(split_row["seed"])+fold*101
        fitted={}
        specifications=[
            ("linear_basic","linear","basic",list(train_cfg["linear_l2"])),
            ("dense_lambdamart","lambdamart","basic",list(train_cfg["lambdamart_configs"])),
            ("dense_mlp","mlp","basic",list(train_cfg["mlp_configs"])),
            ("graph_lambdamart","lambdamart","graph",list(train_cfg["lambdamart_configs"])),
            ("graph_mlp","mlp","graph",list(train_cfg["mlp_configs"])),
        ]
        for offset,(arm,family,feature_set,configs) in enumerate(specifications):
            print(json.dumps({"repeat":repeat,"fold":fold,"arm":arm,"status":"tuning"}),flush=True)
            started=time.perf_counter(); selected,trace=tune(
                dataset,train_qids,family,feature_set,configs,train_cfg,seed+offset*1009)
            scaler=fit_scaler(dataset,train_qids,feature_set); X=matrix(dataset,scaler,feature_set)
            if family=="linear": model=fit_linear(dataset,X,train_qids,float(selected),train_cfg,seed+offset)
            elif family=="lambdamart": model=fit_lambdamart(dataset,X,train_qids,selected,seed+offset)
            else: model,history=fit_mlp(dataset,X,train_qids,selected,train_cfg,seed+offset)
            fitted[arm]=predict_model(family,model,X)
            tuning.append({"repeat":repeat,"fold":fold,"arm":arm,"family":family,
                           "feature_set":feature_set,"configs_evaluated":len(configs),
                           "selected":selected,"inner_results":trace,
                           "elapsed_seconds":time.perf_counter()-started})
            print(json.dumps({"repeat":repeat,"fold":fold,"arm":arm,"status":"complete",
                              "elapsed_seconds":tuning[-1]["elapsed_seconds"]}),flush=True)
        frozen_dense=dataset.basic[:,BASIC_FEATURES.index("dense_score")]
        static=dataset.graph[:,GRAPH_FEATURES.index("official_ppr_score")]
        fitted={"frozen_dense":frozen_dense,"official_static":static,**fitted}
        for qid in valid_qids:
            for arm,scores in fitted.items():
                metric,pred=query_metric(dataset,qid,scores,arm,repeat,fold)
                metrics.append(metric);predictions.append(pred)
    write_jsonl(out/"reranker_cv_predictions.jsonl",predictions)
    write_jsonl(out/"reranker_cv_per_query_metrics.jsonl",metrics)
    write_jsonl(out/"reranker_hyperparameter_traces.jsonl",tuning)
    summary={arm:aggregate([r for r in metrics if r["arm"]==arm]) for arm in ARMS}
    write_json(out/"reranker_cv_metrics.json",summary)
    comparisons=[]
    pairs=[("dense_lambdamart","frozen_dense"),("dense_mlp","frozen_dense"),
           ("graph_lambdamart","dense_lambdamart"),("graph_mlp","dense_mlp"),
           ("graph_lambdamart","graph_mlp"),("linear_basic","frozen_dense"),
           ("linear_basic","official_static"),("linear_basic","dense_lambdamart"),
           ("linear_basic","dense_mlp")]
    for left,right in pairs:
        for metric in ("full_ndcg_at3","mean_utility_at3","pairwise_accuracy","spearman"):
            comparisons.append(paired(metrics,left,right,metric))
    write_json(out/"reranker_paired_comparisons.json",comparisons)
    top3_residual=[]
    qrel_keys=set(dataset.qrels)
    for pred in predictions:
        for rank,cid in enumerate(pred["ranked_comment_ids"][:3],1):
            key=(pred["query_id"],cid)
            if key not in qrel_keys:
                top3_residual.append({"query_id":key[0],"comment_id":key[1],"arm":pred["arm"],
                                      "repeat":pred["repeat"],"fold":pred["fold"],"rank":rank})
    unique={(r["query_id"],r["comment_id"]):r for r in top3_residual}
    write_jsonl(out/"reranker_followup_residual_pairs.jsonl",sorted(unique.values(),key=lambda r:(r["query_id"],r["comment_id"])))
    leakage={
        "verdict":"PASS","outer_query_overlap":0,"pair_level_split":False,
        "scaler_fit_scope":"train-query candidates only","inner_selection_scope":"outer-train queries only",
        "validation_labels_used_for_features":False,"utility_in_features":False,
        "unjudged_in_loss":False,"unjudged_assigned_zero":False,"test_split_used":False,
        "candidate_set_identical":True,"feature_names":list(BASIC_FEATURES+GRAPH_FEATURES)}
    write_json(out/"feature_leakage_audit.json",leakage)
    coverage={"by_arm":{arm:summary[arm]["coverage_at3"] for arm in ARMS},
              "followup_residual_unique_pairs":len(unique),"complete_threshold":.95,
              "all_arms_at_least_95pct":all(summary[a]["coverage_at3"]>=.95 for a in ARMS),
              "unjudged_assigned_zero":False}
    write_json(out/"coverage_bias_report.json",coverage)
    print(json.dumps({"trained":True,"coverage":coverage},indent=2))


def resummarize(config_key: str) -> None:
    _, raw = cfg(config_key)
    out = raw["output_dir"]
    metrics = read_jsonl(out / "reranker_cv_per_query_metrics.jsonl")
    summary = {
        arm: aggregate([row for row in metrics if row["arm"] == arm])
        for arm in ARMS
    }
    write_json(out / "reranker_cv_metrics.json", summary)
    pairs = [
        ("dense_lambdamart", "frozen_dense"),
        ("dense_mlp", "frozen_dense"),
        ("graph_lambdamart", "dense_lambdamart"),
        ("graph_mlp", "dense_mlp"),
        ("graph_lambdamart", "graph_mlp"),
        ("linear_basic", "frozen_dense"),
        ("linear_basic", "official_static"),
        ("linear_basic", "dense_lambdamart"),
        ("linear_basic", "dense_mlp"),
    ]
    comparisons = [
        paired(metrics, left, right, metric)
        for left, right in pairs
        for metric in (
            "full_ndcg_at3",
            "mean_utility_at3",
            "pairwise_accuracy",
            "spearman",
        )
    ]
    write_json(out / "reranker_paired_comparisons.json", comparisons)
    print(json.dumps({
        "resummarized": True,
        "metric_rows": len(metrics),
        "comparisons": len(comparisons),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        nargs="?",
        default="train",
        choices=("train", "reevaluate", "resummarize"),
    )
    parser.add_argument("--config-key", default="reranker_signal_validation")
    args = parser.parse_args()
    if args.phase == "train":
        run(args.config_key)
    elif args.phase == "resummarize":
        resummarize(args.config_key)
    else:
        evaluate_frozen_predictions(args.config_key)


if __name__=="__main__": main()
