"""Stable-ID prefix/future training views from one continuous fitted jewel field."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.streaming import build_rolling_windows, measure_jewel_lifecycles
from sol.streaming_features import to_frontier_time
from sol.token_grid import GridCapacityError, GridSpec


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, sets: list[torch.Tensor]) -> "FeatureStandardizer":
        nonempty = [values.double() for values in sets if len(values)]
        if not nonempty:
            raise ValueError("normalization requires at least one feature")
        values = torch.cat(nonempty)
        return cls(values.mean(0).float(), values.std(0).clamp_min(1e-4).float())

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean.to(features)) / self.std.to(features)

    def denormalize(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.std.to(features) + self.mean.to(features)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "FeatureStandardizer":
        return cls(state["mean"].float(), state["std"].float())


@dataclass(frozen=True)
class BirthTarget:
    values: torch.Tensor
    cell_indices: torch.Tensor
    slot_indices: torch.Tensor
    counts: torch.Tensor
    global_ids: torch.Tensor
    birth_frames: torch.Tensor


@dataclass(frozen=True)
class ContinuationView:
    index: int
    frontier: int
    commit_stop: int
    context_features: torch.Tensor
    context_ids: torch.Tensor
    carried_global_features: torch.Tensor
    carried_ids: torch.Tensor
    births: BirthTarget
    target_active_global_features: torch.Tensor
    active_commit_ids: torch.Tensor


@dataclass(frozen=True)
class ContinuationDataset:
    views: tuple[ContinuationView, ...]
    context_standardizer: FeatureStandardizer
    birth_standardizer: FeatureStandardizer
    grid_spec: GridSpec
    total_frames: int
    prefix_frames: int
    stride_frames: int
    support_sigma: float


def rasterize_context(
    features: torch.Tensor,
    standardizer: FeatureStandardizer,
    *,
    prefix_frames: int,
    stride_frames: int,
    grid_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Pool normalized prefix features into count/mean/variance raster channels."""
    if not len(features):
        raise ValueError("context raster requires at least one jewel")
    gu, gv, gt = grid_shape
    scaled = (features[:, :2].clamp(-1, 1) + 1) * 0.5
    u = (scaled[:, 0] * gu).long().clamp_max(gu - 1)
    v = (scaled[:, 1] * gv).long().clamp_max(gv - 1)
    prefix_strides = prefix_frames / stride_frames
    relative_t = ((features[:, 2] + prefix_strides) / prefix_strides).clamp(0, 1)
    t = (relative_t * gt).long().clamp_max(gt - 1)
    cells = (u * gv + v) * gt + t
    values = standardizer.normalize(features)
    total = values.new_zeros(gu * gv * gt, values.shape[1])
    square = torch.zeros_like(total)
    count = values.new_zeros(gu * gv * gt, 1)
    indices = cells[:, None].expand_as(values)
    total.scatter_add_(0, indices, values)
    square.scatter_add_(0, indices, values.square())
    count.scatter_add_(0, cells[:, None], values.new_ones(len(values), 1))
    mean = total / count.clamp_min(1)
    variance = (square / count.clamp_min(1) - mean.square()).clamp_min(0)
    log_count = torch.log1p(count)
    occupied = (count > 0).to(values.dtype)
    return torch.cat((mean, variance, log_count, occupied), dim=1)


def _birth_cells(
    values: torch.Tensor,
    birth_frames: torch.Tensor,
    frontier: int,
    stride_frames: int,
    spec: GridSpec,
) -> torch.Tensor:
    gu, gv, gt = spec.shape
    scaled = (values[:, :2].clamp(-1, 1) + 1) * 0.5
    u = (scaled[:, 0] * gu).long().clamp_max(gu - 1)
    v = (scaled[:, 1] * gv).long().clamp_max(gv - 1)
    relative = (birth_frames - frontier).clamp(0, stride_frames - 1)
    t = (relative * gt // stride_frames).long().clamp_max(gt - 1)
    return (u * gv + v) * gt + t


def _pack_births(
    values: torch.Tensor,
    global_ids: torch.Tensor,
    birth_frames: torch.Tensor,
    frontier: int,
    stride_frames: int,
    spec: GridSpec,
) -> BirthTarget:
    cells = _birth_cells(values, birth_frames, frontier, stride_frames, spec)
    counts = torch.bincount(cells, minlength=spec.n_cells)
    maximum = int(counts.max()) if len(counts) else 0
    if maximum > spec.slots_per_cell:
        raise GridCapacityError(
            f"birth grid needs {maximum} slots but contract allows {spec.slots_per_cell}"
        )
    order = torch.arange(len(values), device=values.device)
    for key in (global_ids, values[:, 2], values[:, 1], values[:, 0], cells):
        order = order[torch.argsort(key[order], stable=True)]
    cells = cells[order]
    offsets = counts.cumsum(0) - counts
    slots = torch.arange(len(cells), device=values.device) - torch.repeat_interleave(
        offsets, counts
    )
    return BirthTarget(
        values=values[order],
        cell_indices=cells,
        slot_indices=slots,
        counts=counts,
        global_ids=global_ids[order],
        birth_frames=birth_frames[order],
    )


def build_continuation_dataset(
    features: torch.Tensor,
    frames: int,
    *,
    prefix_frames: int = 32,
    stride_frames: int = 16,
    support_sigma: float = 3.0,
    grid_spec: GridSpec = GridSpec((16, 16, 8), 128),
) -> ContinuationDataset:
    """Create canonical continuation pairs and train-only feature standardizers."""
    if prefix_frames <= 0 or prefix_frames + stride_frames > frames:
        raise ValueError("prefix/future windows do not fit inside the clip")
    lifecycles = measure_jewel_lifecycles(
        features, frames, support_sigma=support_sigma
    )
    rolling = build_rolling_windows(
        lifecycles,
        frames,
        prefix_frames=prefix_frames,
        stride_frames=stride_frames,
    )
    views = []
    for window in rolling:
        if window.frontier < prefix_frames:
            continue
        if window.commit_stop - window.frontier != stride_frames:
            continue
        context = to_frontier_time(
            features[window.context_ids], frames, window.frontier, stride_frames
        )
        birth_values = to_frontier_time(
            features[window.birth_ids], frames, window.frontier, stride_frames
        )
        births = _pack_births(
            birth_values,
            window.birth_ids,
            lifecycles.first_active_frames[window.birth_ids],
            window.frontier,
            stride_frames,
            grid_spec,
        )
        views.append(
            ContinuationView(
                index=window.index,
                frontier=window.frontier,
                commit_stop=window.commit_stop,
                context_features=context,
                context_ids=window.context_ids,
                carried_global_features=features[window.carried_ids].clone(),
                carried_ids=window.carried_ids,
                births=births,
                target_active_global_features=features[
                    window.active_commit_ids
                ].clone(),
                active_commit_ids=window.active_commit_ids,
            )
        )
    if not views:
        raise ValueError("no complete continuation views were produced")
    context_standardizer = FeatureStandardizer.fit(
        [view.context_features for view in views]
    )
    birth_standardizer = FeatureStandardizer.fit(
        [view.births.values for view in views]
    )
    return ContinuationDataset(
        views=tuple(views),
        context_standardizer=context_standardizer,
        birth_standardizer=birth_standardizer,
        grid_spec=grid_spec,
        total_frames=frames,
        prefix_frames=prefix_frames,
        stride_frames=stride_frames,
        support_sigma=support_sigma,
    )
