"""Differentiable full-frame appearance objectives and range diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AppearanceObjective:
    """Named components of one spatiotemporal appearance objective."""

    total: torch.Tensor
    rgb: torch.Tensor
    spatial: torch.Tensor
    temporal: torch.Tensor
    structure: torch.Tensor
    range: torch.Tensor


def _validate_video_pair(rendered: torch.Tensor, target: torch.Tensor) -> None:
    if rendered.shape != target.shape or rendered.ndim != 4 or rendered.shape[-1] != 3:
        raise ValueError("rendered and target must share F,H,W,3 video shapes")


def multiscale_charbonnier(
    rendered: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Score RGB disagreement over a three-level area-averaged image pyramid."""
    _validate_video_pair(rendered, target)
    prediction = rendered.permute(0, 3, 1, 2)
    reference = target.permute(0, 3, 1, 2)
    loss = rendered.new_zeros(())
    weight_sum = 0.0
    for weight in (1.0, 0.5, 0.25):
        difference = prediction - reference
        loss = loss + weight * torch.sqrt(difference.square() + 1e-6).mean()
        weight_sum += weight
        if min(prediction.shape[-2:]) < 2:
            break
        prediction = F.avg_pool2d(prediction, 2)
        reference = F.avg_pool2d(reference, 2)
    return loss / weight_sum


def spatial_edge_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compare horizontal and vertical first differences."""
    _validate_video_pair(rendered, target)
    prediction = rendered.permute(0, 3, 1, 2)
    reference = target.permute(0, 3, 1, 2)
    horizontal = F.l1_loss(
        prediction[..., 1:] - prediction[..., :-1],
        reference[..., 1:] - reference[..., :-1],
    )
    vertical = F.l1_loss(
        prediction[..., 1:, :] - prediction[..., :-1, :],
        reference[..., 1:, :] - reference[..., :-1, :],
    )
    return 0.5 * (horizontal + vertical)


def temporal_edge_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compare consecutive-frame RGB changes, returning zero for a single frame."""
    _validate_video_pair(rendered, target)
    if len(rendered) < 2:
        return rendered.new_zeros(())
    return F.l1_loss(torch.diff(rendered, dim=0), torch.diff(target, dim=0))


def spatiotemporal_structure_loss(
    rendered: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return a differentiable global SSIM loss across space and contiguous time."""
    _validate_video_pair(rendered, target)
    dimensions = (0, 1, 2)
    candidate_mean = rendered.mean(dim=dimensions)
    target_mean = target.mean(dim=dimensions)
    candidate_centered = rendered - candidate_mean
    target_centered = target - target_mean
    candidate_variance = candidate_centered.square().mean(dim=dimensions)
    target_variance = target_centered.square().mean(dim=dimensions)
    covariance = (candidate_centered * target_centered).mean(dim=dimensions)
    c1, c2 = 0.01**2, 0.03**2
    luminance = (2 * candidate_mean * target_mean + c1) / (
        candidate_mean.square() + target_mean.square() + c1
    )
    contrast_structure = (2 * covariance + c2) / (
        candidate_variance + target_variance + c2
    )
    return 1 - (luminance * contrast_structure).mean()


def range_excess_loss(values: torch.Tensor) -> torch.Tensor:
    """Penalize squared RGB distance outside the displayable [0,1] interval."""
    if values.numel() == 0:
        raise ValueError("range loss requires at least one value")
    return (F.relu(-values).square() + F.relu(values - 1).square()).mean()


def range_diagnostics(values: torch.Tensor) -> dict[str, torch.Tensor]:
    """Measure below-zero, above-one, and total out-of-range fractions plus excess."""
    if values.numel() == 0:
        raise ValueError("range diagnostics require at least one value")
    below = values < 0
    above = values > 1
    return {
        "below_zero_fraction": below.float().mean(),
        "above_one_fraction": above.float().mean(),
        "out_of_range_fraction": (below | above).float().mean(),
        "range_excess": range_excess_loss(values),
    }


def residual_energy(residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean-square residual RGB and RGB-Jacobian energy separately."""
    if residual.ndim != 2 or residual.shape[1] != 12:
        raise ValueError("appearance residual must have shape N,12")
    return residual[:, :3].square().mean(), residual[:, 3:].square().mean()


def appearance_objective(
    rendered: torch.Tensor,
    target: torch.Tensor,
    *,
    rgb_weight: float = 1.0,
    spatial_weight: float = 0.5,
    temporal_weight: float = 0.0,
    structure_weight: float = 0.0,
    range_weight: float = 0.0,
) -> AppearanceObjective:
    """Combine named video terms without hiding their individually auditable scales."""
    weights = (
        rgb_weight,
        spatial_weight,
        temporal_weight,
        structure_weight,
        range_weight,
    )
    if any(weight < 0 for weight in weights) or not any(weights):
        raise ValueError("appearance weights must be non-negative and not all zero")
    rgb = multiscale_charbonnier(rendered, target)
    spatial = spatial_edge_loss(rendered, target)
    temporal = temporal_edge_loss(rendered, target)
    structure = spatiotemporal_structure_loss(rendered, target)
    range_loss = range_excess_loss(rendered)
    total = (
        rgb_weight * rgb
        + spatial_weight * spatial
        + temporal_weight * temporal
        + structure_weight * structure
        + range_weight * range_loss
    )
    return AppearanceObjective(
        total=total,
        rgb=rgb,
        spatial=spatial,
        temporal=temporal,
        structure=structure,
        range=range_loss,
    )


def multiscale_image_loss(
    rendered: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Preserve the original RGB-pyramid plus spatial-edge objective exactly."""
    return appearance_objective(rendered, target).total
