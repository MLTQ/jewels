"""Occupancy-aware raster token indexing, statistics, and lossless packing."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class GridCapacityError(ValueError):
    """Raised instead of silently dropping jewels from an overfull cell."""


@dataclass(frozen=True)
class GridSpec:
    shape: tuple[int, int, int] = (8, 8, 4)
    slots_per_cell: int = 256

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(value <= 0 for value in self.shape):
            raise ValueError("shape must contain three positive cell counts")
        if self.slots_per_cell <= 0:
            raise ValueError("slots_per_cell must be positive")

    @property
    def n_cells(self) -> int:
        gu, gv, gt = self.shape
        return gu * gv * gt

    @property
    def max_jewels(self) -> int:
        return self.n_cells * self.slots_per_cell

    def cell_index(self, centers: torch.Tensor) -> torch.Tensor:
        """Map normalized centers shaped (...,3) to raster-ordered cell IDs."""
        gu, gv, gt = self.shape
        scaled = (centers.clamp(-1, 1) + 1) * 0.5
        u = (scaled[..., 0] * gu).long().clamp_max(gu - 1)
        v = (scaled[..., 1] * gv).long().clamp_max(gv - 1)
        t = (scaled[..., 2] * gt).long().clamp_max(gt - 1)
        return (u * gv + v) * gt + t

    def cells_for_aabb(
        self,
        minimum: torch.Tensor,
        maximum: torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Conservatively mark every cell touched by a normalized world AABB."""
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("minimum and maximum must have shape (3,)")
        if (minimum > maximum).any():
            raise ValueError("minimum must not exceed maximum")
        target_device = device if device is not None else minimum.device
        gu, gv, gt = self.shape
        cell_min = torch.tensor([-1.0, -1.0, -1.0], device=target_device)
        cell_size = torch.tensor(
            [2.0 / gu, 2.0 / gv, 2.0 / gt], device=target_device
        )
        ids = torch.arange(self.n_cells, device=target_device)
        t = ids % gt
        v = (ids // gt) % gv
        u = ids // (gv * gt)
        coords = torch.stack([u, v, t], dim=-1).to(cell_size.dtype)
        lo = cell_min + coords * cell_size
        hi = lo + cell_size
        query_lo = minimum.to(device=target_device, dtype=lo.dtype)
        query_hi = maximum.to(device=target_device, dtype=lo.dtype)
        return ((hi >= query_lo) & (lo <= query_hi)).all(dim=-1)


@dataclass(frozen=True)
class CapacityReport:
    total_jewels: int
    n_cells: int
    slots_per_cell: int
    max_cell_occupancy: int
    overflow_cells: int

    @property
    def fits(self) -> bool:
        return self.overflow_cells == 0


@dataclass
class PackedGrid:
    values: torch.Tensor
    mask: torch.Tensor
    counts: torch.Tensor


@dataclass
class CompactGrid:
    values: torch.Tensor
    cell_indices: torch.Tensor
    slot_indices: torch.Tensor
    counts: torch.Tensor


@dataclass
class GridStatistics:
    count: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor


def _canonical_order(features: torch.Tensor) -> torch.Tensor:
    """Stable lexicographic order by center u, then v, then t."""
    order = torch.arange(features.shape[0], device=features.device)
    for dimension in (2, 1, 0):
        rank = torch.argsort(features[order, dimension], stable=True)
        order = order[rank]
    return order


class OccupancyGrid:
    """Lossless set-to-cell packing under an explicit capacity contract."""

    def __init__(self, spec: GridSpec = GridSpec()) -> None:
        self.spec = spec

    def capacity_report(self, features: torch.Tensor) -> CapacityReport:
        batched = features if features.ndim == 3 else features[None]
        if batched.ndim != 3 or batched.shape[-1] < 3:
            raise ValueError("features must have shape (N,F) or (B,N,F)")
        indices = self.spec.cell_index(batched[..., :3])
        counts = torch.stack(
            [torch.bincount(row, minlength=self.spec.n_cells) for row in indices]
        )
        return CapacityReport(
            total_jewels=int(batched.shape[0] * batched.shape[1]),
            n_cells=self.spec.n_cells,
            slots_per_cell=self.spec.slots_per_cell,
            max_cell_occupancy=int(counts.max()),
            overflow_cells=int((counts > self.spec.slots_per_cell).sum()),
        )

    def pack(self, features: torch.Tensor) -> PackedGrid:
        """Pack canonical cell slots; raise on overflow instead of truncating."""
        squeeze = features.ndim == 2
        batched = features[None] if squeeze else features
        compact = self.pack_compact(batched)
        batch, _, feature_dim = batched.shape
        values = batched.new_zeros(
            batch, self.spec.n_cells, self.spec.slots_per_cell, feature_dim
        )
        mask = torch.zeros(
            batch,
            self.spec.n_cells,
            self.spec.slots_per_cell,
            dtype=torch.bool,
            device=batched.device,
        )
        for batch_index in range(batch):
            cells = compact.cell_indices[batch_index]
            slots = compact.slot_indices[batch_index]
            values[batch_index, cells, slots] = compact.values[batch_index]
            mask[batch_index, cells, slots] = True
        if squeeze:
            return PackedGrid(values[0], mask[0], compact.counts[0])
        return PackedGrid(values, mask, compact.counts)

    def pack_compact(self, features: torch.Tensor) -> CompactGrid:
        """Pack only occupied canonical slots for memory-efficient training."""
        squeeze = features.ndim == 2
        batched = features[None] if squeeze else features
        if batched.ndim != 3 or batched.shape[-1] < 3:
            raise ValueError("features must have shape (N,F) or (B,N,F)")
        if not torch.isfinite(batched).all():
            raise ValueError("features must be finite")
        report = self.capacity_report(batched)
        if not report.fits:
            raise GridCapacityError(
                f"{report.overflow_cells} cells exceed {report.slots_per_cell} slots; "
                f"maximum occupancy is {report.max_cell_occupancy}"
            )
        indices = self.spec.cell_index(batched[..., :3])
        values_out, cells_out, slots_out, counts_out = [], [], [], []
        for batch_index in range(batched.shape[0]):
            counts = torch.bincount(
                indices[batch_index], minlength=self.spec.n_cells
            )
            order = _canonical_order(batched[batch_index])
            cell_rank = torch.argsort(indices[batch_index, order], stable=True)
            order = order[cell_rank]
            cells = indices[batch_index, order]
            offsets = counts.cumsum(0) - counts
            slots = torch.arange(
                batched.shape[1], device=batched.device
            ) - torch.repeat_interleave(offsets, counts)
            values_out.append(batched[batch_index, order])
            cells_out.append(cells)
            slots_out.append(slots)
            counts_out.append(counts)
        compact = CompactGrid(
            values=torch.stack(values_out),
            cell_indices=torch.stack(cells_out).long(),
            slot_indices=torch.stack(slots_out).long(),
            counts=torch.stack(counts_out),
        )
        if squeeze:
            return CompactGrid(
                compact.values[0],
                compact.cell_indices[0],
                compact.slot_indices[0],
                compact.counts[0],
            )
        return compact

    def statistics(self, features: torch.Tensor) -> GridStatistics:
        """Return per-cell count, feature mean, and feature variance."""
        squeeze = features.ndim == 2
        batched = features[None] if squeeze else features
        if batched.ndim != 3:
            raise ValueError("features must have shape (N,F) or (B,N,F)")
        batch, _, feature_dim = batched.shape
        indices = self.spec.cell_index(batched[..., :3])
        total = batched.new_zeros(batch, self.spec.n_cells, feature_dim)
        square = torch.zeros_like(total)
        count = batched.new_zeros(batch, self.spec.n_cells, 1)
        total.scatter_add_(1, indices[..., None].expand_as(batched), batched)
        square.scatter_add_(1, indices[..., None].expand_as(batched), batched.square())
        count.scatter_add_(
            1,
            indices[..., None],
            torch.ones_like(indices, dtype=batched.dtype)[..., None],
        )
        mean = total / count.clamp_min(1)
        variance = (square / count.clamp_min(1) - mean.square()).clamp_min(0)
        if squeeze:
            return GridStatistics(count[0, :, 0], mean[0], variance[0])
        return GridStatistics(count[..., 0], mean, variance)
