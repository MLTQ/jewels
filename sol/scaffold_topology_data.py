"""Sequential scaffold-topology targets and carried-state rasters."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.splat_density import temporal_standard_deviation
from sol.streaming import build_rolling_windows, measure_jewel_lifecycles
from sol.streaming_data import BirthTarget, pack_births
from sol.streaming_features import to_frontier_time
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class ScaffoldTopologyView:
    """One complete emission stride, including the initial stride at frontier zero."""

    index: int
    frontier: int
    commit_stop: int
    context_features: torch.Tensor
    context_ids: torch.Tensor
    carried_global_features: torch.Tensor
    carried_ids: torch.Tensor
    births: BirthTarget
    birth_global_features: torch.Tensor
    target_active_global_features: torch.Tensor
    active_commit_ids: torch.Tensor


def build_scaffold_topology_views(
    features: torch.Tensor,
    frames: int,
    *,
    stride_frames: int = 16,
    support_sigma: float = 3.0,
    grid_spec: GridSpec = GridSpec((16, 16, 8), 1024),
) -> tuple[ScaffoldTopologyView, ...]:
    """Build initial and continuation topology targets from one persistent field."""
    if stride_frames <= 0 or stride_frames > frames:
        raise ValueError("stride_frames must fit inside the clip")
    lifecycles = measure_jewel_lifecycles(
        features, frames, support_sigma=support_sigma
    )
    windows = build_rolling_windows(
        lifecycles,
        frames,
        prefix_frames=stride_frames,
        stride_frames=stride_frames,
    )
    views = []
    for window in windows:
        if window.commit_stop - window.frontier != stride_frames:
            continue
        local_births = to_frontier_time(
            features[window.birth_ids], frames, window.frontier, stride_frames
        )
        births = pack_births(
            local_births,
            window.birth_ids,
            lifecycles.first_active_frames[window.birth_ids],
            window.frontier,
            stride_frames,
            grid_spec,
        )
        views.append(
            ScaffoldTopologyView(
                index=window.index,
                frontier=window.frontier,
                commit_stop=window.commit_stop,
                context_features=to_frontier_time(
                    features[window.context_ids],
                    frames,
                    window.frontier,
                    stride_frames,
                ),
                context_ids=window.context_ids,
                carried_global_features=features[window.carried_ids].clone(),
                carried_ids=window.carried_ids,
                births=births,
                birth_global_features=features[births.global_ids].clone(),
                target_active_global_features=features[
                    window.active_commit_ids
                ].clone(),
                active_commit_ids=window.active_commit_ids,
            )
        )
    if not views or views[0].frontier != 0:
        raise ValueError("topology dataset requires one complete initial stride")
    return tuple(views)


def rasterize_carried_state(
    features: torch.Tensor,
    total_frames: int,
    frontier: int,
    stride_frames: int,
    spec: GridSpec,
    *,
    support_sigma: float = 3.0,
) -> torch.Tensor:
    """Rasterize immutable carried jewels as log-density, occupancy, and mean alpha."""
    if features.ndim != 2 or features.shape[1] != 22:
        raise ValueError("features must have shape (jewels, 22)")
    if support_sigma <= 0:
        raise ValueError("support_sigma must be positive")
    if not len(features):
        return features.new_zeros(spec.n_cells, 3)

    local = to_frontier_time(features, total_frames, frontier, stride_frames)
    gu, gv, gt = spec.shape
    scaled = (local[:, :2].clamp(-1, 1) + 1) * 0.5
    u = (scaled[:, 0] * gu).long().clamp_max(gu - 1)
    v = (scaled[:, 1] * gv).long().clamp_max(gv - 1)
    spatial = u * gv + v

    times = (torch.arange(gt, device=features.device, dtype=features.dtype) + 0.5) / gt
    sigma = temporal_standard_deviation(local).clamp_min(1e-8)
    standardized = (times[:, None] - local[None, :, 2]).abs() / sigma[None]
    active = standardized <= support_sigma
    alpha = torch.sigmoid(local[None, :, 21]) * torch.exp(
        -0.5 * standardized.square()
    )
    cells = spatial[None] * gt + torch.arange(
        gt, device=features.device
    )[:, None]
    active_cells = cells[active]
    count = features.new_zeros(spec.n_cells)
    alpha_sum = features.new_zeros(spec.n_cells)
    count.scatter_add_(0, active_cells, features.new_ones(len(active_cells)))
    alpha_sum.scatter_add_(0, active_cells, alpha[active])
    log_density = torch.log1p(count) / torch.log(
        count.new_tensor(float(spec.slots_per_cell + 1))
    )
    occupied = (count > 0).to(features.dtype)
    mean_alpha = alpha_sum / count.clamp_min(1)
    return torch.stack((log_density, occupied, mean_alpha), dim=1)
