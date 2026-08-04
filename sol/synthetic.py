"""Deterministic synthetic canonical jewel features for spike tests."""

from __future__ import annotations

import math

import torch


def random_jewels(
    count: int,
    *,
    seed: int = 0,
    scale: float = 0.08,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create valid `(N,22)` canonical features with uniform centers."""
    if count <= 0:
        raise ValueError("count must be positive")
    generator = torch.Generator(device=device).manual_seed(seed)
    features = torch.zeros(count, 22, device=device)
    features[:, :3] = torch.rand(count, 3, generator=generator, device=device) * 2 - 1
    log_variance = 2 * math.log(scale)
    features[:, 3] = log_variance
    features[:, 6] = log_variance
    features[:, 8] = log_variance
    features[:, 9:12] = torch.rand(
        count, 3, generator=generator, device=device
    )
    features[:, 21] = -1.0
    return features


def elongated_knn_counterexample() -> tuple[torch.Tensor, torch.Tensor]:
    """Return jewels/point where 64 nearer centers hide a broad contributor."""
    features = random_jewels(65, seed=1, scale=0.01)
    features[:64, :3] = torch.stack(
        [torch.linspace(0.05, 0.95, 64), torch.zeros(64), torch.zeros(64)], dim=1
    )
    features[:64, 9:12] = 0
    features[:64, 21] = 10
    features[64, :3] = torch.tensor([1.0, 0.0, 0.0])
    log_variance = 2 * math.log(2.0)
    features[64, 3] = log_variance
    features[64, 6] = log_variance
    features[64, 8] = log_variance
    features[64, 9:12] = 1
    features[64, 21] = 10
    return features, torch.zeros(1, 3)
