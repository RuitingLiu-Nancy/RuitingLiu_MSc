"""Query-balanced training and evaluation for the LUAD seed pilot."""
from __future__ import annotations

import copy
import math
import random
import statistics
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr

from evaluation.ir_metrics import graded_ndcg_at

from ..config import LUADConfig
from ..data.judgment_dataset import JudgmentDataset
from ..models.seed_scorer import CandidateMLP, LearnedSeedOnly
from .losses import eligible_pairs, query_ranking_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def build_model(name: str, sample: dict, cfg: LUADConfig) -> torch.nn.Module:
    if name == "dense_mlp":
        return CandidateMLP((0, 3), cfg.hidden_dim)
    if name == "graph_feature_mlp":
        return CandidateMLP((0, 2, 3, 4, 5, 6), cfg.hidden_dim)
    if name in {"learned_seed_pure", "learned_seed_head"}:
        return LearnedSeedOnly(
            seed_feature_dim=sample["candidate_seed_features"].shape[1],
            candidate_feature_dim=sample["candidate_features"].shape[1],
            hidden_dim=cfg.hidden_dim, damping=cfg.damping, steps=cfg.local_steps,
            ranking_head=name.endswith("head"),
        )
    raise ValueError(name)


def train_fold(name: str, dataset: JudgmentDataset, train_qids: list[str],
               cfg: LUADConfig, seed: int) -> tuple[torch.nn.Module, dict]:
    seed_everything(seed)
    sample = dataset.load(train_qids[0])
    model = build_model(name, sample, cfg)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best_state, best_loss, stale = copy.deepcopy(model.state_dict()), float("inf"), 0
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        optimizer.zero_grad()
        query_losses, pair_count = [], 0
        order = list(train_qids)
        random.Random(seed + epoch).shuffle(order)
        for qid in order:
            graph = dataset.load(qid)
            scores, _ = model(graph)
            loss, detail = query_ranking_loss(
                scores, graph["utility"], graph["judged_mask"],
                margin=cfg.pair_margin, max_pairs=cfg.max_pairs_per_query,
                lambda_regression=cfg.lambda_regression, generator=generator)
            if detail["pairs"]:
                query_losses.append(loss)
                pair_count += detail["pairs"]
        if not query_losses:
            raise RuntimeError("no eligible within-query ranking pairs")
        loss = torch.stack(query_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        current = float(loss.detach())
        history.append({"epoch": epoch + 1, "train_query_mean_loss": current,
                        "sampled_pairs": pair_count})
        if current < best_loss - 1e-5:
            best_loss, stale = current, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, {
        "epochs_run": len(history), "best_train_query_mean_loss": best_loss,
        "first_train_query_mean_loss": history[0]["train_query_mean_loss"],
        "last_train_query_mean_loss": history[-1]["train_query_mean_loss"],
        "early_stopping_signal": "training-loss plateau only; validation labels not used",
        "history": history,
    }


def static_scores(graph: dict, name: str) -> torch.Tensor:
    feature = {"dense": 0, "official_static": 2}[name]
    return graph["candidate_features"][:, feature].float()


def query_metrics(qid: str, graph: dict, scores: torch.Tensor, arm: str) -> tuple[dict, dict]:
    values = scores.detach().cpu().numpy()
    order = np.argsort(-values, kind="stable")
    judged = graph["judged_mask"].numpy().astype(bool)
    utility = graph["utility"].numpy()
    comments = graph["candidate_comment_ids"]
    full_top3 = order[:3]
    full_judged = [int(i) for i in full_top3 if judged[i]]
    judged_order = [int(i) for i in order if judged[i]]
    gains = {comments[i]: float(utility[i]) for i in np.flatnonzero(judged)}
    ranked_judged = [comments[i] for i in judged_order]
    observed = [float(utility[i]) for i in full_judged]

    judged_indices = torch.nonzero(graph["judged_mask"], as_tuple=False).flatten()
    generator = torch.Generator().manual_seed(0)
    pairs = eligible_pairs(graph["utility"], graph["judged_mask"], margin=1e-9,
                           max_pairs=10**9, generator=generator)
    pair_acc = (float((scores[pairs[:, 0]] > scores[pairs[:, 1]]).float().mean())
                if len(pairs) else float("nan"))
    rho = spearmanr(values[judged], utility[judged]).statistic if len(judged_indices) > 1 else float("nan")
    row = {
        "query_id": qid, "arm": arm, "candidate_count": len(comments),
        "judged_count": int(judged.sum()),
        "full_ranking_judgment_coverage_at3": len(full_judged) / 3.0,
        "observed_mean_utility_at3": statistics.fmean(observed) if observed else None,
        "observed_utility_count_at3": len(observed),
        "judged_only_mean_utility_at3": statistics.fmean(
            float(utility[i]) for i in judged_order[:3]),
        "judged_only_ndcg_at3": graded_ndcg_at(ranked_judged, gains, 3),
        "pairwise_accuracy": pair_acc, "spearman": float(rho),
    }
    prediction = {
        "query_id": qid, "arm": arm,
        "ranked_comment_ids": [comments[int(i)] for i in order],
        "scores": [float(values[int(i)]) for i in order],
        "judged_mask_in_rank_order": [bool(judged[int(i)]) for i in order],
    }
    return row, prediction


def summarize_metrics(rows: list[dict]) -> dict:
    keys = (
        "full_ranking_judgment_coverage_at3", "observed_mean_utility_at3",
        "judged_only_mean_utility_at3", "judged_only_ndcg_at3",
        "pairwise_accuracy", "spearman",
    )
    out = {"query_evaluations": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows
                  if row[key] is not None and math.isfinite(float(row[key]))]
        out[key] = statistics.fmean(values) if values else None
        out[f"{key}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return out


def run_cross_validation(dataset: JudgmentDataset, splits: list[dict], cfg: LUADConfig,
                         eligible_queries: set[str]) -> tuple[dict, list[dict], list[dict], list[dict]]:
    arms = ("dense", "official_static", "dense_mlp", "graph_feature_mlp",
            "learned_seed_pure", "learned_seed_head")
    per_query, predictions, training = [], [], []
    for split in splits:
        train_qids = [qid for qid in split["train_query_ids"] if qid in eligible_queries]
        valid_qids = [qid for qid in split["validation_query_ids"] if qid in eligible_queries]
        if not valid_qids:
            continue
        models = {}
        for offset, name in enumerate(arms[2:]):
            model, trace = train_fold(
                name, dataset, train_qids, cfg,
                seed=int(split["seed"]) + int(split["fold"]) * 101 + offset)
            models[name] = model
            training.append({
                "repeat": split["repeat"], "fold": split["fold"], "arm": name,
                "train_queries": len(train_qids), "validation_queries": len(valid_qids),
                **trace,
            })
        for qid in valid_qids:
            graph = dataset.load(qid)
            for name in arms:
                if name in {"dense", "official_static"}:
                    score = static_scores(graph, name)
                else:
                    models[name].eval()
                    with torch.no_grad():
                        score, aux = models[name](graph)
                metric, pred = query_metrics(qid, graph, score, name)
                metric.update({"repeat": split["repeat"], "fold": split["fold"]})
                pred.update({"repeat": split["repeat"], "fold": split["fold"]})
                if name.startswith("learned_seed"):
                    distribution = aux["seed_distribution"].detach().cpu().numpy()
                    order = np.argsort(-distribution)
                    pred["learned_seed_distribution"] = [
                        {
                            **graph["candidate_seed_metadata"][int(index)],
                            "learned_probability": float(distribution[int(index)]),
                        }
                        for index in order
                    ]
                per_query.append(metric)
                predictions.append(pred)

    by_arm = defaultdict(list)
    for row in per_query:
        by_arm[row["arm"]].append(row)
    summary = {arm: summarize_metrics(by_arm[arm]) for arm in arms}
    return summary, per_query, predictions, training


def toy_overfit(dataset: JudgmentDataset, qid: str, cfg: LUADConfig) -> dict:
    seed_everything(cfg.seeds[0])
    graph = dataset.load(qid)
    model = build_model("learned_seed_head", graph, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=max(cfg.learning_rate, .01))
    generator = torch.Generator().manual_seed(cfg.seeds[0])
    first, last = None, None
    for _ in range(300):
        optimizer.zero_grad()
        scores, _ = model(graph)
        loss, _ = query_ranking_loss(
            scores, graph["utility"], graph["judged_mask"], margin=cfg.pair_margin,
            max_pairs=10**6, lambda_regression=cfg.lambda_regression, generator=generator)
        first = float(loss.detach()) if first is None else first
        loss.backward(); optimizer.step()
        last = float(loss.detach())
    model.eval()
    with torch.no_grad():
        score, aux = model(graph)
    metric, _ = query_metrics(qid, graph, score, "toy_learned_seed_head")
    distribution = aux["seed_distribution"].numpy()
    metadata = graph["candidate_seed_metadata"]
    seed_order = np.argsort(-distribution)
    return {
        "query_id": qid, "first_loss": first, "final_loss": last,
        "loss_reduction": first - last, "pairwise_accuracy": metric["pairwise_accuracy"],
        "judged_only_ndcg_at3": metric["judged_only_ndcg_at3"],
        "top_learned_seeds": [
            {**metadata[int(index)], "learned_probability": float(distribution[int(index)])}
            for index in seed_order[:10]
        ],
    }
