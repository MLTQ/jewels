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
    device: torch.device,
    generator: torch.Generator | None,
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
    generator: torch.Generator | None = None,
) -> RealizerRenderLoss:
    """Compare future birth renders on fresh aligned spatiotemporal patches."""
    if predicted_local.shape != target_local.shape or predicted_local.ndim != 2:
        raise ValueError("predicted and target local marks must have matching matrices")
    if predicted_local.shape[1] != 22 or carried_global.ndim != 2:
        raise ValueError("render fields must use canonical 22-D jewel features")
    weights = (rgb_weight, edge_weight, chroma_weight, structure_weight)
    if any(weight < 0 for weight in weights) or not any(weights):
        raise ValueError("render loss weights must be non-negative and not all zero")
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
        device=predicted_local.device,
        generator=generator,
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
    total = (
        rgb_weight * rgb
        + edge_weight * edge
        + chroma_weight * chroma
        + structure_weight * structure
    )
    return RealizerRenderLoss(total, rgb, edge, chroma, structure)
