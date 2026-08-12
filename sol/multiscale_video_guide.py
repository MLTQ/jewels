"""Multiscale local video tokens aligned with the jewel birth grid."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from sol.token_grid import GridSpec


MULTISCALE_GUIDE_FEATURE_DIM = 16


def _difference(values: torch.Tensor, dimension: int) -> torch.Tensor:
    """Return bounded first differences without wrapping volume boundaries."""
    if values.shape[dimension] == 1:
        return torch.zeros_like(values)
    output = torch.zeros_like(values)
    middle = [slice(None)] * values.ndim
    before = [slice(None)] * values.ndim
    after = [slice(None)] * values.ndim
    middle[dimension] = slice(1, -1)
    before[dimension] = slice(0, -2)
    after[dimension] = slice(2, None)
    output[tuple(middle)] = 0.5 * (
        values[tuple(after)] - values[tuple(before)]
    )
    first = [slice(None)] * values.ndim
    second = [slice(None)] * values.ndim
    first[dimension] = 0
    second[dimension] = 1
    output[tuple(first)] = values[tuple(second)] - values[tuple(first)]
    last = [slice(None)] * values.ndim
    penultimate = [slice(None)] * values.ndim
    last[dimension] = -1
    penultimate[dimension] = -2
    output[tuple(last)] = values[tuple(last)] - values[tuple(penultimate)]
    return output


def _cell_tokens(
    features: torch.Tensor,
    spec: GridSpec,
    subgrid: tuple[int, int, int],
) -> torch.Tensor:
    """Group dense ``(C,T,V,U)`` samples into canonical ``(u,v,t)`` cells."""
    gu, gv, gt = spec.shape
    su, sv, st = subgrid
    channels = features.shape[0]
    grouped = features.reshape(channels, gt, st, gv, sv, gu, su)
    return grouped.permute(5, 3, 1, 6, 4, 2, 0).reshape(
        spec.n_cells, su * sv * st, channels
    )


def video_to_multiscale_cell_tokens(
    video: torch.Tensor,
    spec: GridSpec,
    *,
    scales: tuple[int, ...] = (1, 2, 4),
    subgrid: tuple[int, int, int] = (2, 2, 2),
) -> torch.Tensor:
    """Encode a ``(T,H,W,3)`` video as aligned local multiscale cell tokens.

    Every scale contributes one RGB, spatial-gradient, temporal-gradient, local-offset,
    and scale-aware token for each location in the declared per-cell subgrid. The output
    order follows :class:`GridSpec`: cells flatten as ``(u,v,t)`` and tokens flatten as
    ``(scale,local-u,local-v,local-t)``.
    """
    if video.ndim != 4 or video.shape[-1] != 3 or min(video.shape[:3]) <= 0:
        raise ValueError("video must have non-empty shape (T,H,W,3)")
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("guide scales must contain positive integers")
    if len(set(scales)) != len(scales):
        raise ValueError("guide scales must be unique")
    if len(subgrid) != 3 or any(size <= 0 for size in subgrid):
        raise ValueError("guide subgrid must contain three positive sizes")

    gu, gv, gt = spec.shape
    su, sv, st = subgrid
    dense_shape = (gt * st, gv * sv, gu * su)
    volume = video.float().permute(3, 0, 1, 2)[None]
    offset_u = (torch.arange(su, device=video.device) + 0.5) * (2 / su) - 1
    offset_v = (torch.arange(sv, device=video.device) + 0.5) * (2 / sv) - 1
    offset_t = (torch.arange(st, device=video.device) + 0.5) * (2 / st) - 1
    local_offsets = torch.stack(
        torch.meshgrid(offset_u, offset_v, offset_t, indexing="ij"), dim=-1
    ).reshape(1, su * sv * st, 3)
    scale_denominator = max(math.log2(max(scales)), 1.0)
    tokens = []
    for scale in scales:
        if scale == 1:
            reduced = volume
        else:
            reduced_shape = tuple(
                max(1, math.ceil(length / scale)) for length in video.shape[:3]
            )
            reduced = F.interpolate(volume, size=reduced_shape, mode="area")
        sampled = F.interpolate(
            reduced, size=dense_shape, mode="trilinear", align_corners=False
        )[0]
        temporal = _difference(sampled, 1)
        vertical = _difference(sampled, 2)
        horizontal = _difference(sampled, 3)
        appearance = _cell_tokens(
            torch.cat((sampled, horizontal, vertical, temporal), dim=0),
            spec,
            subgrid,
        )
        offsets = local_offsets.to(appearance).expand(spec.n_cells, -1, -1)
        scale_value = math.log2(scale) / scale_denominator
        scale_feature = appearance.new_full(
            (spec.n_cells, su * sv * st, 1), scale_value
        )
        tokens.append(torch.cat((appearance, offsets, scale_feature), dim=-1))
    output = torch.cat(tokens, dim=1)
    if output.shape[-1] != MULTISCALE_GUIDE_FEATURE_DIM:
        raise AssertionError("multiscale guide feature contract changed")
    return output
