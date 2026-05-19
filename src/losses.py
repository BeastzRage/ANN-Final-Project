"""
loss helpers for Part B.
"""

import torch


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(embeddings)


def triplet_margin_loss(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    positive_distance = torch.sum((anchor_embeddings - positive_embeddings) ** 2, dim=1)
    negative_distance = torch.sum((anchor_embeddings - negative_embeddings) ** 2, dim=1)
    return torch.relu(positive_distance - negative_distance + eps).mean()
