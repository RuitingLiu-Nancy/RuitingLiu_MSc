"""Discrete personalized propagation adapted from PPNP/APPNP.

Klicpera et al.'s power iteration is retained, but local transition
normalization is row-stochastic weighted random walk (matching HippoRAG's
undirected personalized PageRank) instead of APPNP's symmetric GCN matrix.
No PyG dependency is introduced because this project does not otherwise use
PyG and its current environment is incompatible with the historical repos.
"""
from __future__ import annotations

import torch
from torch import nn


def row_normalized_weights(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                           node_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index.shape[0] != 2 or edge_index.shape[1] != edge_weight.numel():
        raise ValueError("edge_index/edge_weight mismatch")
    source = edge_index[0]
    outgoing = torch.zeros(node_count, dtype=edge_weight.dtype, device=edge_weight.device)
    outgoing.index_add_(0, source, edge_weight)
    normalized = edge_weight / outgoing[source].clamp_min(torch.finfo(edge_weight.dtype).eps)
    return normalized, outgoing


def personalized_power_iteration(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    restart: torch.Tensor,
    *,
    damping: float,
    steps: int,
) -> torch.Tensor:
    """Differentiable K-step PPR with restart-aware dangling-node handling."""
    if not (0.0 < damping < 1.0) or steps < 1:
        raise ValueError("invalid propagation controls")
    if restart.ndim != 1 or torch.any(restart < 0) or restart.sum() <= 0:
        raise ValueError("restart must be a non-negative vector with positive mass")
    restart = restart / restart.sum()
    node_count = restart.numel()
    normalized, outgoing = row_normalized_weights(edge_index, edge_weight, node_count)
    source, target = edge_index
    probability = restart
    for _ in range(steps):
        propagated = torch.zeros_like(probability)
        propagated.index_add_(0, target, normalized * probability[source])
        dangling_mass = probability[outgoing == 0].sum()
        propagated = propagated + dangling_mass * restart
        probability = damping * propagated + (1.0 - damping) * restart
    return probability / probability.sum().clamp_min(torch.finfo(probability.dtype).eps)


@torch.no_grad()
def seed_to_candidate_kernel(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    seed_nodes: torch.Tensor,
    candidate_nodes: torch.Tensor,
    *,
    node_count: int,
    damping: float,
    steps: int,
) -> torch.Tensor:
    """Precompute the linear fixed-graph map from each seed to candidates.

    Seed Only never learns edges, so propagation is linear in restart mass.  This
    exact batched basis avoids repeating the same sparse walk in every epoch;
    gradients still flow through the subsequent kernel/distribution product.
    """
    seed_count = int(seed_nodes.numel())
    restart = torch.zeros((node_count, seed_count), dtype=edge_weight.dtype)
    restart[seed_nodes, torch.arange(seed_count)] = 1.0
    normalized, outgoing = row_normalized_weights(edge_index, edge_weight, node_count)
    source, target = edge_index
    probability = restart
    for _ in range(steps):
        propagated = torch.zeros_like(probability)
        propagated.index_add_(0, target, normalized[:, None] * probability[source])
        dangling_mass = probability[outgoing == 0].sum(dim=0, keepdim=True)
        propagated = propagated + restart * dangling_mass
        probability = damping * propagated + (1.0 - damping) * restart
    return probability[candidate_nodes]


class StaticAPPNP(nn.Module):
    """Parameter-free wrapper used by static reproduction and seed models."""

    def __init__(self, damping: float, steps: int):
        super().__init__()
        self.damping = float(damping)
        self.steps = int(steps)

    def forward(self, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                restart: torch.Tensor) -> torch.Tensor:
        return personalized_power_iteration(
            edge_index, edge_weight, restart, damping=self.damping, steps=self.steps)
