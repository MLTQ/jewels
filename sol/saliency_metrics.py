"""Label-free foreground, motion-boundary, and temporal-stability video metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SaliencyRenderSignature:
    foreground_rgb_mae: float
    foreground_psnr: float
    foreground_edge_mae: float
    motion_boundary_mae: float
    quiet_temporal_mae: float


def _spatial_score(values: torch.Tensor) -> torch.Tensor:
    score = values.new_zeros(values.shape[:-1])
    if values.shape[1] > 1:
        vertical = torch.diff(values, dim=1).abs().mean(dim=-1)
        score[:, :-1] += vertical
        score[:, 1:] += vertical
    if values.shape[2] > 1:
        horizontal = torch.diff(values, dim=2).abs().mean(dim=-1)
        score[:, :, :-1] += horizontal
        score[:, :, 1:] += horizontal
    return score


def _spatial_error(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = candidate.new_zeros(candidate.shape[:-1])
    if candidate.shape[1] > 1:
        vertical = (
            torch.diff(candidate, dim=1) - torch.diff(target, dim=1)
        ).abs().mean(dim=-1)
        error[:, :-1] += vertical
        error[:, 1:] += vertical
    if candidate.shape[2] > 1:
        horizontal = (
            torch.diff(candidate, dim=2) - torch.diff(target, dim=2)
        ).abs().mean(dim=-1)
        error[:, :, :-1] += horizontal
        error[:, :, 1:] += horizontal
    return error


def _fraction_mask(
    scores: torch.Tensor, fraction: float, *, largest: bool
) -> torch.Tensor:
    if not 0 < fraction <= 1:
        raise ValueError("mask fraction must lie inside (0,1]")
    flat = scores.flatten()
    count = max(1, round(len(flat) * fraction))
    selected = flat.topk(count, largest=largest).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[selected] = True
    return mask.reshape_as(scores)


def saliency_render_signature(
    candidate: torch.Tensor,
    target: torch.Tensor,
    *,
    background: torch.Tensor,
    foreground_fraction: float = 0.2,
    motion_fraction: float = 0.2,
    quiet_fraction: float = 0.5,
) -> SaliencyRenderSignature:
    """Measure errors on target-derived salient and temporally quiet regions."""
    if candidate.shape != target.shape or candidate.ndim != 4:
        raise ValueError("candidate and target must share shape (T,H,W,3)")
    if candidate.shape[-1] != 3 or len(candidate) < 2:
        raise ValueError("saliency metrics require at least two RGB frames")
    if background.shape != (3,):
        raise ValueError("background must have shape (3,)")
    if not torch.isfinite(candidate).all() or not torch.isfinite(target).all():
        raise ValueError("candidate and target must be finite")

    target = target.float()
    candidate = candidate.float()
    chroma = target.amax(dim=-1) - target.amin(dim=-1)
    foreground_score = (
        (target - background.float()).abs().mean(dim=-1)
        + 0.5 * chroma
        + _spatial_score(target)
    )
    foreground = _fraction_mask(
        foreground_score, foreground_fraction, largest=True
    )
    pixel_error = (candidate - target).abs().mean(dim=-1)
    foreground_mae = pixel_error[foreground].mean()
    foreground_mse = (
        (candidate - target).square().mean(dim=-1)[foreground].mean()
    )

    edge_error = _spatial_error(candidate, target)[foreground].mean()

    candidate_change = torch.diff(candidate, dim=0)
    target_change = torch.diff(target, dim=0)
    temporal_error = (candidate_change - target_change).abs().mean(dim=-1)
    motion_magnitude = target_change.abs().mean(dim=-1)
    motion_score = motion_magnitude + _spatial_score(target_change)
    motion_boundary = _fraction_mask(motion_score, motion_fraction, largest=True)
    quiet = _fraction_mask(motion_score, quiet_fraction, largest=False)
    psnr = min(100.0, -10 * math.log10(max(float(foreground_mse), 1e-10)))
    return SaliencyRenderSignature(
        foreground_rgb_mae=float(foreground_mae),
        foreground_psnr=psnr,
        foreground_edge_mae=float(edge_error),
        motion_boundary_mae=float(temporal_error[motion_boundary].mean()),
        quiet_temporal_mae=float(temporal_error[quiet].mean()),
    )
