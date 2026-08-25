"""Differentiable perceptual supervision for train-only appearance renders."""

from __future__ import annotations

import torch


def build_lpips_training_metric(device: torch.device, net: str = "alex"):
    """Build a frozen LPIPS network while preserving gradients to its input."""
    import lpips  # noqa: PLC0415

    metric = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    metric.requires_grad_(False)
    return metric


def perceptual_training_loss(
    candidate: torch.Tensor,
    target: torch.Tensor,
    metric,
) -> torch.Tensor:
    """Score matching FHWC videos through an injected differentiable metric."""
    if candidate.shape != target.shape or candidate.ndim != 4 or candidate.shape[-1] != 3:
        raise ValueError("candidate and target must share shape (F,H,W,3)")
    candidate_nchw = candidate.clamp(0.0, 1.0).permute(0, 3, 1, 2)
    target_nchw = target.clamp(0.0, 1.0).permute(0, 3, 1, 2)
    return metric(candidate_nchw, target_nchw, normalize=True).mean()
