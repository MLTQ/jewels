"""Multilevel center bins for support-complete spacetime rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


class SupportOverflowError(RuntimeError):
    """Raised when a support-safe query exceeds its declared candidate budget."""


@dataclass(frozen=True)
class TileLevel:
    """Sorted primitive-center bins for one conservative support-radius level."""

    cell_width: float
    sorted_keys: torch.Tensor
    sorted_primitive_indices: torch.Tensor


@dataclass(frozen=True)
class SupportTileIndex:
    """Detached multilevel index rebuilt from the current field geometry."""

    mu: torch.Tensor
    max_scale: torch.Tensor
    half_extent: torch.Tensor | None
    metric_scale: torch.Tensor | None
    metric_rotation: torch.Tensor | None
    support_sigma: float
    levels: tuple[TileLevel, ...]


_COORD_BITS = 21
_COORD_BIAS = 1 << (_COORD_BITS - 1)
_COORD_MAX = _COORD_BIAS - 1
_NEIGHBOR_OFFSETS = torch.tensor(
    [
        (x, y, z)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
    ],
    dtype=torch.long,
)


def _pack_cells(cells: torch.Tensor) -> torch.Tensor:
    """Pack signed integer xyz cells into collision-free positive int64 keys."""
    if cells.shape[-1] != 3:
        raise ValueError("cells must end in xyz coordinates")
    if cells.numel() == 0:
        return torch.empty(cells.shape[:-1], dtype=torch.long, device=cells.device)
    if bool(((cells < -_COORD_BIAS) | (cells > _COORD_MAX)).any()):
        raise ValueError("tile coordinate exceeds signed 21-bit packing range")
    biased = cells + _COORD_BIAS
    return (
        (biased[..., 0] << (2 * _COORD_BITS))
        | (biased[..., 1] << _COORD_BITS)
        | biased[..., 2]
    )


def build_support_tile_index(
    mu: torch.Tensor,
    max_scale: torch.Tensor,
    *,
    half_extent: torch.Tensor | None = None,
    metric_scale: torch.Tensor | None = None,
    metric_rotation: torch.Tensor | None = None,
    support_sigma: float = 5.0,
    base_resolution: int = 32,
    level_scale: float = 1.55,
) -> SupportTileIndex:
    """Assign every primitive center to one support-bound-matched cell."""
    if mu.ndim != 2 or mu.shape[1] != 3:
        raise ValueError("mu must have shape (num_primitives, 3)")
    if max_scale.shape != (mu.shape[0],):
        raise ValueError("max_scale must have shape (num_primitives,)")
    if half_extent is not None and half_extent.shape != mu.shape:
        raise ValueError("half_extent must have shape (num_primitives, 3)")
    if (metric_scale is None) != (metric_rotation is None):
        raise ValueError("metric_scale and metric_rotation must be supplied together")
    if metric_scale is not None and metric_scale.shape != mu.shape:
        raise ValueError("metric_scale must have shape (num_primitives, 3)")
    if metric_rotation is not None and metric_rotation.shape != (
        mu.shape[0], 3, 3
    ):
        raise ValueError("metric_rotation must have shape (num_primitives, 3, 3)")
    if mu.shape[0] == 0:
        raise ValueError("cannot index an empty primitive field")
    if support_sigma <= 0:
        raise ValueError("support_sigma must be positive")
    if base_resolution <= 0:
        raise ValueError("base_resolution must be positive")
    if level_scale <= 1.0:
        raise ValueError("level_scale must be greater than one")

    mu = mu.detach()
    max_scale = max_scale.detach().clamp_min(1e-8)
    if half_extent is not None:
        half_extent = half_extent.detach().clamp_min(1e-8)
        radius = half_extent.amax(dim=1)
    else:
        radius = support_sigma * max_scale
    base_width = 2.0 / base_resolution
    log_scale = math.log(level_scale)
    level = torch.ceil(torch.log(radius / base_width) / log_scale).clamp_min(0)
    level = level.to(torch.long)
    width = base_width * torch.pow(
        torch.tensor(level_scale, dtype=radius.dtype, device=radius.device),
        level,
    )
    # Floating exponentiation can undershoot an exact boundary by one ulp. A
    # too-small cell would invalidate the neighbor proof, so advance it.
    level = level + (width < radius).to(torch.long)

    groups = []
    for level_value in torch.unique(level, sorted=True).tolist():
        primitive_indices = torch.where(level == level_value)[0]
        cell_width = base_width * (level_scale**level_value)
        cells = torch.floor((mu[primitive_indices] + 1.0) / cell_width).long()
        keys = _pack_cells(cells)
        sorted_keys, order = keys.sort()
        groups.append(
            TileLevel(
                cell_width=cell_width,
                sorted_keys=sorted_keys,
                sorted_primitive_indices=primitive_indices[order],
            )
        )
    return SupportTileIndex(
        mu=mu,
        max_scale=max_scale,
        half_extent=half_extent,
        metric_scale=(metric_scale.detach() if metric_scale is not None else None),
        metric_rotation=(
            metric_rotation.detach() if metric_rotation is not None else None
        ),
        support_sigma=support_sigma,
        levels=tuple(groups),
    )


def _expand_sorted_ranges(
    starts: torch.Tensor,
    counts: torch.Tensor,
    owners: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand searchsorted ranges into parallel owner and sorted-position rows."""
    nonempty = counts > 0
    starts = starts[nonempty]
    counts = counts[nonempty]
    owners = owners[nonempty]
    total = int(counts.sum())
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=starts.device)
        return empty, empty
    expanded_owners = torch.repeat_interleave(owners, counts)
    block_starts = torch.cumsum(counts, dim=0) - counts
    within = torch.arange(total, device=starts.device) - torch.repeat_interleave(
        block_starts, counts
    )
    positions = torch.repeat_interleave(starts, counts) + within
    return expanded_owners, positions


def query_support_pairs(
    index: SupportTileIndex,
    points: torch.Tensor,
    *,
    capacity: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return complete conservative-sphere (query, primitive) candidate pairs."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (num_points, 3)")
    if capacity <= 0:
        raise ValueError("support capacity must be positive")
    if points.shape[0] == 0:
        empty = torch.empty(0, dtype=torch.long, device=points.device)
        return empty, empty

    neighbor_offsets = _NEIGHBOR_OFFSETS.to(points.device)
    point_template = torch.arange(points.shape[0], device=points.device)
    point_template = point_template.repeat_interleave(len(neighbor_offsets))
    owner_parts = []
    primitive_parts = []
    for group in index.levels:
        point_cells = torch.floor((points + 1.0) / group.cell_width).long()
        neighbor_cells = point_cells[:, None, :] + neighbor_offsets[None, :, :]
        query_keys = _pack_cells(neighbor_cells).reshape(-1)
        starts = torch.searchsorted(group.sorted_keys, query_keys, right=False)
        ends = torch.searchsorted(group.sorted_keys, query_keys, right=True)
        owners, positions = _expand_sorted_ranges(
            starts, ends - starts, point_template
        )
        if positions.numel():
            owner_parts.append(owners)
            primitive_parts.append(group.sorted_primitive_indices[positions])

    if not owner_parts:
        empty = torch.empty(0, dtype=torch.long, device=points.device)
        return empty, empty
    owners = torch.cat(owner_parts)
    primitive_indices = torch.cat(primitive_parts)

    displacement = points[owners] - index.mu[primitive_indices]
    if index.half_extent is None:
        radius = index.support_sigma * index.max_scale[primitive_indices]
        inside_bound = displacement.square().sum(dim=1) <= radius.square()
    else:
        extent = index.half_extent[primitive_indices]
        inside_bound = (displacement.abs() <= extent).all(dim=1)
    owners = owners[inside_bound]
    primitive_indices = primitive_indices[inside_bound]

    if index.metric_rotation is not None and owners.numel():
        displacement = points[owners] - index.mu[primitive_indices]
        rotated = torch.einsum(
            "pji,pj->pi", index.metric_rotation[primitive_indices], displacement
        )
        normalized = rotated / (index.metric_scale[primitive_indices] + 1e-8)
        inside_ellipsoid = normalized.square().sum(dim=1) <= (
            index.support_sigma * index.support_sigma
        )
        owners = owners[inside_ellipsoid]
        primitive_indices = primitive_indices[inside_ellipsoid]

    counts = torch.bincount(owners, minlength=points.shape[0])
    overflow = counts > capacity
    if bool(overflow.any()):
        raise SupportOverflowError(
            f"support candidate capacity {capacity} is insufficient for "
            f"{int(overflow.sum())}/{points.shape[0]} query points; increase "
            "support_capacity or reduce support_sigma"
        )
    return owners, primitive_indices
