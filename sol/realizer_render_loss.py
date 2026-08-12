"""Differentiable local render supervision for video-to-jewel realization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sol.render import render_exact


@dataclass(frozen=True)
class RealizerRenderLoss:
    total: torch.Tensor
    rgb: torch.Tensor
    edge: torch.Tensor
    chroma: torch.Tensor
    structure: torch.Tensor
    saliency_rgb: torch.Tensor
    motion: torch.Tensor
    stability: torch.Tensor


def estimate_target_marks(
    noised_values: torch.Tensor,
    predicted_velocity: torch.Tensor,
    flow_time: torch.Tensor,
) -> torch.Tensor:
    """Estimate the clean endpoint of a rectified-flow path at ``flow_time``."""
    if noised_values.shape != predicted_velocity.shape:
        raise ValueError("noised values and velocity must have matching shapes")
    if flow_time.ndim == 0:
        flow_time = flow_time[None]
    if flow_time.shape != (1,):
        raise ValueError("flow time must contain one value")
    return noised_values + (1 - flow_time) * predicted_velocity


def _sample_patch_points(
    *,
    total_frames: int,
    frontier: int,
    stride_frames: int,
    render_height: int,
    render_width: int,
    patches: int,
    patch_frames: int,
    patch_height: int,
    patch_width: int,
    anchor_frontier: bool,
    device: torch.device,
    generator: torch.Generator | None,
    patch_importance: torch.Tensor | None = None,
    importance_grid_shape: tuple[int, int, int] | None = None,
    importance_fraction: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if total_frames < 2 or not 0 <= frontier < total_frames:
        raise ValueError("invalid fitted-video time contract")
    if frontier + stride_frames > total_frames:
        raise ValueError("future stride exceeds fitted video")
    if min(render_height, render_width, patches, patch_frames, patch_height, patch_width) <= 0:
        raise ValueError("render and patch dimensions must be positive")
    if patch_frames > stride_frames:
        raise ValueError("patch frames exceed the future stride")
    if patch_height > render_height or patch_width > render_width:
        raise ValueError("patch dimensions exceed the render grid")
    if not 0 <= importance_fraction <= 1:
        raise ValueError("importance fraction must be in [0,1]")
    if importance_fraction and (
        patch_importance is None or importance_grid_shape is None
    ):
        raise ValueError("importance sampling requires weights and a grid shape")

    frame_start = torch.randint(
        stride_frames - patch_frames + 1,
        (patches,),
        device=device,
        generator=generator,
    )
    row_start = torch.randint(
        render_height - patch_height + 1,
        (patches,),
        device=device,
        generator=generator,
    )
    column_start = torch.randint(
        render_width - patch_width + 1,
        (patches,),
        device=device,
        generator=generator,
    )
    if importance_fraction:
        assert patch_importance is not None
        assert importance_grid_shape is not None
        gu, gv, gt = importance_grid_shape
        if min(importance_grid_shape) <= 0 or patch_importance.shape != (
            gu * gv * gt,
        ):
            raise ValueError("patch importance does not match its grid")
        weights = patch_importance.to(device=device, dtype=torch.float32)
        if (
            not torch.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError(
                "patch importance must be finite, non-negative, and non-empty"
            )
        importance_patches = max(1, round(patches * importance_fraction))
        importance_patches = min(patches, importance_patches)
        selected = torch.multinomial(
            weights,
            importance_patches,
            replacement=True,
            generator=generator,
        )
        selected_t = selected % gt
        selected_v = (selected // gt) % gv
        selected_u = selected // (gv * gt)
        frame_center = ((2 * selected_t + 1) * stride_frames) // (2 * gt)
        row_center = ((2 * selected_v + 1) * render_height) // (2 * gv)
        column_center = ((2 * selected_u + 1) * render_width) // (2 * gu)
        frame_start[:importance_patches] = (
            frame_center - patch_frames // 2
        ).clamp(0, stride_frames - patch_frames)
        row_start[:importance_patches] = (
            row_center - patch_height // 2
        ).clamp(0, render_height - patch_height)
        column_start[:importance_patches] = (
            column_center - patch_width // 2
        ).clamp(0, render_width - patch_width)
    if anchor_frontier:
        frame_start[0] = 0
    frame_offset = torch.arange(patch_frames, device=device)
    row_offset = torch.arange(patch_height, device=device)
    column_offset = torch.arange(patch_width, device=device)
    frames = frame_start[:, None, None, None] + frame_offset[None, :, None, None]
    rows = row_start[:, None, None, None] + row_offset[None, None, :, None]
    columns = (
        column_start[:, None, None, None] + column_offset[None, None, None, :]
    )
    frames = frames.expand(-1, -1, patch_height, patch_width)
    rows = rows.expand(-1, patch_frames, -1, patch_width)
    columns = columns.expand(-1, patch_frames, patch_height, -1)
    absolute_frames = frames + frontier
    u = (columns.float() + 0.5) * (2 / render_width) - 1
    v = (rows.float() + 0.5) * (2 / render_height) - 1
    global_t = absolute_frames.float() * (2 / (total_frames - 1)) - 1
    local_t = frames.float() / stride_frames
    global_points = torch.stack((u, v, global_t), dim=-1).reshape(-1, 3)
    local_points = torch.stack((u, v, local_t), dim=-1).reshape(-1, 3)
    return global_points, local_points


def _normalize_importance(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean()
    return values / mean.clamp_min(1e-4) if float(mean) > 0 else values


def scaffold_saliency_weights(
    guide_raster: torch.Tensor,
    grid_shape: tuple[int, int, int],
    background: torch.Tensor,
) -> torch.Tensor:
    """Score scaffold cells by foreground, motion, chroma, and spatial boundaries."""
    gu, gv, gt = grid_shape
    if min(grid_shape) <= 0 or guide_raster.shape != (gu * gv * gt, 3):
        raise ValueError("guide raster does not match the saliency grid")
    if background.shape != (3,):
        raise ValueError("background must have shape (3,)")
    volume = guide_raster.reshape(gu, gv, gt, 3).permute(2, 1, 0, 3).float()
    foreground = (volume - background.float()).abs().mean(dim=-1)
    chroma = volume.amax(dim=-1) - volume.amin(dim=-1)
    motion = torch.zeros_like(foreground)
    if gt > 1:
        change = (volume[1:] - volume[:-1]).abs().mean(dim=-1)
        motion[:-1] += change
        motion[1:] += change
    boundary = torch.zeros_like(foreground)
    if gu > 1:
        horizontal = (volume[:, :, 1:] - volume[:, :, :-1]).abs().mean(dim=-1)
        boundary[:, :, :-1] += horizontal
        boundary[:, :, 1:] += horizontal
    if gv > 1:
        vertical = (volume[:, 1:] - volume[:, :-1]).abs().mean(dim=-1)
        boundary[:, :-1] += vertical
        boundary[:, 1:] += vertical
    weights = (
        0.25
        + _normalize_importance(foreground)
        + 2 * _normalize_importance(motion)
        + 0.5 * _normalize_importance(chroma)
        + _normalize_importance(boundary)
    )
    return weights.permute(2, 1, 0).reshape(-1)


def _edge_loss(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for dimension in (1, 2, 3):
        if candidate.shape[dimension] > 1:
            losses.append(
                F.l1_loss(
                    torch.diff(candidate, dim=dimension),
                    torch.diff(target, dim=dimension),
                )
            )
    return torch.stack(losses).mean() if losses else candidate.new_zeros(())


def _chroma(values: torch.Tensor) -> torch.Tensor:
    red, green, blue = values.unbind(dim=-1)
    return torch.stack((red - green, blue - 0.5 * (red + green)), dim=-1)


def _structure_loss(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a differentiable patchwise SSIM loss over space and time."""
    dimensions = (1, 2, 3)
    candidate_mean = candidate.mean(dim=dimensions)
    target_mean = target.mean(dim=dimensions)
    candidate_centered = candidate - candidate_mean[:, None, None, None]
    target_centered = target - target_mean[:, None, None, None]
    candidate_variance = candidate_centered.square().mean(dim=dimensions)
    target_variance = target_centered.square().mean(dim=dimensions)
    covariance = (candidate_centered * target_centered).mean(dim=dimensions)
    c1 = 0.01**2
    c2 = 0.03**2
    luminance = (
        2 * candidate_mean * target_mean + c1
    ) / (candidate_mean.square() + target_mean.square() + c1)
    contrast_structure = (2 * covariance + c2) / (
        candidate_variance + target_variance + c2
    )
    return 1 - (luminance * contrast_structure).mean()


def _render_saliency(expected: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    foreground = (expected - background).abs().mean(dim=-1)
    chroma = expected.amax(dim=-1) - expected.amin(dim=-1)
    boundary = torch.zeros_like(foreground)
    if expected.shape[2] > 1:
        vertical = (expected[:, :, 1:] - expected[:, :, :-1]).abs().mean(dim=-1)
        boundary[:, :, :-1] += vertical
        boundary[:, :, 1:] += vertical
    if expected.shape[3] > 1:
        horizontal = (expected[:, :, :, 1:] - expected[:, :, :, :-1]).abs().mean(
            dim=-1
        )
        boundary[:, :, :, :-1] += horizontal
        boundary[:, :, :, 1:] += horizontal
    weights = (
        0.25
        + _normalize_importance(foreground)
        + 0.5 * _normalize_importance(chroma)
        + _normalize_importance(boundary)
    )
    return weights / weights.mean().clamp_min(1e-4)


def _motion_losses(
    candidate: torch.Tensor, expected: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidate.shape[1] < 2:
        zero = candidate.new_zeros(())
        return zero, zero
    candidate_delta = torch.diff(candidate, dim=1)
    expected_delta = torch.diff(expected, dim=1)
    error = (candidate_delta - expected_delta).abs().mean(dim=-1)
    magnitude = expected_delta.abs().mean(dim=-1)
    boundary = torch.zeros_like(magnitude)
    if magnitude.shape[2] > 1:
        vertical = (magnitude[:, :, 1:] - magnitude[:, :, :-1]).abs()
        boundary[:, :, :-1] += vertical
        boundary[:, :, 1:] += vertical
    if magnitude.shape[3] > 1:
        horizontal = (magnitude[:, :, :, 1:] - magnitude[:, :, :, :-1]).abs()
        boundary[:, :, :, :-1] += horizontal
        boundary[:, :, :, 1:] += horizontal
    motion_weight = (
        0.25
        + _normalize_importance(magnitude)
        + _normalize_importance(boundary)
    )
    motion_weight = motion_weight / motion_weight.mean().clamp_min(1e-4)
    motion = (error * motion_weight).mean()
    quiet_weight = torch.exp(-magnitude / magnitude.mean().clamp_min(1e-3))
    quiet_weight = quiet_weight / quiet_weight.mean().clamp_min(1e-4)
    stability = (error * quiet_weight).mean()
    return motion, stability


def realizer_render_loss(
    predicted_local: torch.Tensor,
    target_local: torch.Tensor,
    carried_global: torch.Tensor,
    *,
    total_frames: int,
    frontier: int,
    stride_frames: int,
    background: torch.Tensor,
    render_height: int = 24,
    render_width: int = 40,
    patches: int = 2,
    patch_frames: int = 2,
    patch_height: int = 4,
    patch_width: int = 4,
    rgb_weight: float = 1.0,
    edge_weight: float = 0.25,
    chroma_weight: float = 0.25,
    structure_weight: float = 0.25,
    saliency_weight: float = 0.0,
    motion_weight: float = 0.0,
    stability_weight: float = 0.0,
    guide_raster: torch.Tensor | None = None,
    guide_grid_shape: tuple[int, int, int] | None = None,
    saliency_fraction: float = 0.0,
    anchor_frontier: bool = False,
    generator: torch.Generator | None = None,
) -> RealizerRenderLoss:
    """Compare future birth renders on fresh aligned spatiotemporal patches."""
    if predicted_local.shape != target_local.shape or predicted_local.ndim != 2:
        raise ValueError("predicted and target local marks must have matching matrices")
    if predicted_local.shape[1] != 22 or carried_global.ndim != 2:
        raise ValueError("render fields must use canonical 22-D jewel features")
    weights = (
        rgb_weight,
        edge_weight,
        chroma_weight,
        structure_weight,
        saliency_weight,
        motion_weight,
        stability_weight,
    )
    if any(weight < 0 for weight in weights) or not any(weights):
        raise ValueError("render loss weights must be non-negative and not all zero")
    patch_importance = None
    if saliency_fraction:
        if guide_raster is None or guide_grid_shape is None:
            raise ValueError("saliency sampling requires a guide raster and grid shape")
        patch_importance = scaffold_saliency_weights(
            guide_raster, guide_grid_shape, background
        )
    global_points, local_points = _sample_patch_points(
        total_frames=total_frames,
        frontier=frontier,
        stride_frames=stride_frames,
        render_height=render_height,
        render_width=render_width,
        patches=patches,
        patch_frames=patch_frames,
        patch_height=patch_height,
        patch_width=patch_width,
        anchor_frontier=anchor_frontier,
        device=predicted_local.device,
        generator=generator,
        patch_importance=patch_importance,
        importance_grid_shape=guide_grid_shape,
        importance_fraction=saliency_fraction,
    )
    shape = (patches, patch_frames, patch_height, patch_width, 3)
    with torch.no_grad():
        base = render_exact(
            carried_global.float(), global_points, background=background.float()
        )
        expected = (
            base + render_exact(target_local.float(), local_points)
        ).clamp(0, 1).reshape(shape)
    candidate = (
        base + render_exact(predicted_local.float(), local_points)
    ).clamp(0, 1).reshape(shape)
    rgb = F.mse_loss(candidate, expected)
    edge = _edge_loss(candidate, expected)
    chroma = F.l1_loss(_chroma(candidate), _chroma(expected))
    structure = _structure_loss(candidate, expected)
    saliency = (
        (candidate - expected).square().mean(dim=-1)
        * _render_saliency(expected, background.float())
    ).mean()
    motion, stability = _motion_losses(candidate, expected)
    total = (
        rgb_weight * rgb
        + edge_weight * edge
        + chroma_weight * chroma
        + structure_weight * structure
        + saliency_weight * saliency
        + motion_weight * motion
        + stability_weight * stability
    )
    return RealizerRenderLoss(
        total, rgb, edge, chroma, structure, saliency, motion, stability
    )
