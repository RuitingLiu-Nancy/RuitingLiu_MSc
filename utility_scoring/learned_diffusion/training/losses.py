"""Query-balanced silver-label objectives for LUAD."""
from __future__ import annotations

import torch
import torch.nn.functional as functional


def eligible_pairs(utilities: torch.Tensor, judged_mask: torch.Tensor, *,
                   margin: float, max_pairs: int, generator: torch.Generator) -> torch.Tensor:
    judged = torch.nonzero(judged_mask, as_tuple=False).flatten()
    if judged.numel() < 2:
        return torch.empty((0, 2), dtype=torch.long)
    left = judged[:, None].expand(-1, len(judged)).reshape(-1)
    right = judged[None, :].expand(len(judged), -1).reshape(-1)
    keep = utilities[left] - utilities[right] >= margin
    pairs = torch.stack((left[keep], right[keep]), dim=1)
    if len(pairs) > max_pairs:
        order = torch.randperm(len(pairs), generator=generator)[:max_pairs]
        pairs = pairs[order]
    return pairs


def query_ranking_loss(scores: torch.Tensor, utilities: torch.Tensor,
                       judged_mask: torch.Tensor, *, margin: float,
                       max_pairs: int, lambda_regression: float,
                       generator: torch.Generator) -> tuple[torch.Tensor, dict]:
    pairs = eligible_pairs(utilities, judged_mask, margin=margin,
                           max_pairs=max_pairs, generator=generator)
    if not len(pairs):
        return scores.sum() * 0.0, {"pairs": 0, "rank": 0.0, "regression": 0.0}
    differences = scores[pairs[:, 0]] - scores[pairs[:, 1]]
    rank_loss = functional.softplus(-differences).mean()
    judged_scores = scores[judged_mask]
    judged_utility = (utilities[judged_mask] - 1.0) / 6.0
    normalized_score = torch.sigmoid(judged_scores)
    regression = functional.huber_loss(normalized_score, judged_utility)
    total = rank_loss + float(lambda_regression) * regression
    return total, {
        "pairs": int(len(pairs)), "rank": float(rank_loss.detach()),
        "regression": float(regression.detach()),
    }
