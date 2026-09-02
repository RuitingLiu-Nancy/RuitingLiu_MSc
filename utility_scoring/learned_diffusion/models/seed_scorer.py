"""Learned Seed Only and matched non-graph baselines."""
from __future__ import annotations

import torch
from torch import nn

from .appnp_static import StaticAPPNP, seed_to_candidate_kernel


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class LearnedSeedOnly(nn.Module):
    """Learn seed restart mass; optionally calibrate comment mass with a head."""

    def __init__(self, seed_feature_dim: int, candidate_feature_dim: int,
                 hidden_dim: int, damping: float, steps: int, *, ranking_head: bool):
        super().__init__()
        self.seed_scorer = MLP(seed_feature_dim, hidden_dim)
        self.propagation = StaticAPPNP(damping=damping, steps=steps)
        self.ranking_head_enabled = bool(ranking_head)
        self.ranking_head = MLP(candidate_feature_dim + 1, hidden_dim) if ranking_head else None

    def forward(self, graph: dict) -> tuple[torch.Tensor, dict]:
        seed_features = graph["candidate_seed_features"].float()
        seed_nodes = graph["candidate_seed_local_nodes"].long()
        if seed_nodes.numel() == 0:
            raise ValueError("LearnedSeedOnly requires at least one candidate seed")
        seed_logits = self.seed_scorer(seed_features).squeeze(-1)
        distribution = torch.softmax(seed_logits, dim=0)
        cache_key = ("seed_to_candidate_kernel", self.propagation.damping,
                     self.propagation.steps)
        if cache_key not in graph:
            graph[cache_key] = seed_to_candidate_kernel(
                graph["edge_index"].long(), graph["edge_weight"].float(), seed_nodes,
                graph["candidate_comment_local_nodes"].long(),
                node_count=len(graph["global_nodes"]), damping=self.propagation.damping,
                steps=self.propagation.steps)
        comment_mass = graph[cache_key] @ distribution
        pure_score = torch.log(comment_mass.clamp_min(1e-12))
        if self.ranking_head is None:
            score = pure_score
        else:
            score = self.ranking_head(torch.cat(
                (pure_score[:, None], graph["candidate_features"].float()), dim=1)).squeeze(-1)
        return score, {
            "seed_distribution": distribution, "comment_mass": comment_mass,
            "pure_log_mass": pure_score, "final_probability": None,
        }


class CandidateMLP(nn.Module):
    """Matched dense-only or static graph-feature MLP without propagation."""

    def __init__(self, feature_indices: tuple[int, ...], hidden_dim: int):
        super().__init__()
        self.feature_indices = feature_indices
        self.scorer = MLP(len(feature_indices), hidden_dim)

    def forward(self, graph: dict) -> tuple[torch.Tensor, dict]:
        features = graph["candidate_features"].float()[:, self.feature_indices]
        return self.scorer(features).squeeze(-1), {}
