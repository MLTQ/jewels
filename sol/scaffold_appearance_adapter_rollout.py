"""Autonomous paired rollout for a frozen mark flow and RGB-only adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from sol.birth_mark_flow import BirthMarkFlowModel, project_birth_topology
from sol.scaffold_appearance_adapter import (
    NON_RGB_DIMENSIONS,
    RGB_DIMENSIONS,
    ScaffoldAppearanceAdapter,
    sample_appearance_adapted_birth_marks,
)
from sol.scaffold_mark_data import generated_window_state, rasterize_scaffold_context
from sol.scaffold_mark_rollout import ScaffoldMarkRollout, ScaffoldMarkWindowReport
from sol.scaffold_topology import ScaffoldTopologyModel
from sol.scaffold_topology_realizer import predict_realizer_topology
from sol.streaming_data import FeatureStandardizer
from sol.streaming_features import to_frontier_time, to_global_time
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class AppearanceAdapterWindowReport:
    """Exact feature ownership and RGB residual audit for one generated stride."""

    index: int
    born_jewels: int
    max_normalized_non_rgb_error: float
    max_local_non_rgb_error: float
    max_global_non_rgb_error: float
    rgb_mae: float
    mean_appearance_strength: float
    active_appearance_fraction: float

    @property
    def non_appearance_exact(self) -> bool:
        return (
            self.max_normalized_non_rgb_error == 0.0
            and self.max_local_non_rgb_error == 0.0
            and self.max_global_non_rgb_error == 0.0
        )

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "born_jewels": self.born_jewels,
            "non_appearance_exact": self.non_appearance_exact,
            "max_normalized_non_rgb_error": self.max_normalized_non_rgb_error,
            "max_local_non_rgb_error": self.max_local_non_rgb_error,
            "max_global_non_rgb_error": self.max_global_non_rgb_error,
            "rgb_mae": self.rgb_mae,
            "mean_appearance_strength": self.mean_appearance_strength,
            "active_appearance_fraction": self.active_appearance_fraction,
        }


@dataclass(frozen=True)
class AppearanceAdapterRollout:
    """Frozen base and RGB-adapted fields sharing one topology and ID sequence."""

    base: ScaffoldMarkRollout
    appearance: ScaffoldMarkRollout
    windows: tuple[AppearanceAdapterWindowReport, ...]

    @property
    def non_appearance_exact(self) -> bool:
        return torch.equal(
            self.base.features[:, NON_RGB_DIMENSIONS],
            self.appearance.features[:, NON_RGB_DIMENSIONS],
        ) and all(window.non_appearance_exact for window in self.windows)

    @property
    def lifecycle_exact(self) -> bool:
        lifecycle = (2, 5, 7, 8)
        return torch.equal(
            self.base.features[:, lifecycle], self.appearance.features[:, lifecycle]
        )

    @property
    def stable_ids_exact(self) -> bool:
        return torch.equal(self.base.global_ids, self.appearance.global_ids)

    @property
    def topology_exact(self) -> bool:
        return len(self.base.counts) == len(self.appearance.counts) and all(
            torch.equal(base, appearance)
            for base, appearance in zip(self.base.counts, self.appearance.counts)
        )

    @property
    def report(self) -> dict:
        return {
            "appearance_dimensions": list(RGB_DIMENSIONS),
            "non_appearance_dimensions": list(NON_RGB_DIMENSIONS),
            "non_appearance_exact": self.non_appearance_exact,
            "lifecycle_exact": self.lifecycle_exact,
            "stable_ids_exact": self.stable_ids_exact,
            "topology_exact": self.topology_exact,
            "base": self.base.report,
            "appearance": self.appearance.report,
            "windows": [window.to_dict() for window in self.windows],
        }


def _maximum_non_rgb_error(base: torch.Tensor, appearance: torch.Tensor) -> float:
    if not len(base):
        return 0.0
    return float(
        (base[:, NON_RGB_DIMENSIONS] - appearance[:, NON_RGB_DIMENSIONS])
        .abs()
        .max()
    )


def _rgb_mae(base: torch.Tensor, appearance: torch.Tensor) -> float:
    if not len(base):
        return 0.0
    return float(
        (base[:, RGB_DIMENSIONS] - appearance[:, RGB_DIMENSIONS]).abs().mean()
    )


@torch.no_grad()
def rollout_scaffold_appearance_adapter(
    topology_model: ScaffoldTopologyModel,
    base_flow: BirthMarkFlowModel,
    adapter: ScaffoldAppearanceAdapter,
    guides: Sequence[torch.Tensor],
    text_condition: torch.Tensor,
    context_standardizer: FeatureStandardizer,
    birth_standardizer: FeatureStandardizer,
    *,
    total_frames: int,
    stride_frames: int,
    support_sigma: float,
    topology_spec: GridSpec,
    occupancy_threshold: float,
    device: str | torch.device,
    steps: int,
    generator: torch.Generator,
    allow_initial_prefrontier: bool = True,
    appearance_cell_weights: Sequence[torch.Tensor] | None = None,
) -> AppearanceAdapterRollout:
    """Generate matched fields while the frozen base owns every non-RGB value."""
    if not guides or len(guides) * stride_frames > total_frames:
        raise ValueError("guides must describe one or more complete strides")
    if base_flow.grid_spec != adapter.grid_spec:
        raise ValueError("base flow and appearance adapter use different grids")
    if topology_spec.shape != base_flow.grid_spec.shape:
        raise ValueError("topology and mark flow use different grid shapes")
    if base_flow.grid_spec.slots_per_cell < topology_spec.slots_per_cell:
        raise ValueError("base flow cannot realize the topology rank capacity")
    if appearance_cell_weights is not None and len(appearance_cell_weights) != len(
        guides
    ):
        raise ValueError("appearance cell weights must align with scaffold strides")
    target_device = torch.device(device)
    topology_model.eval()
    base_flow.eval()
    adapter.eval()
    base_features = torch.empty(0, 22)
    appearance_features = torch.empty(0, 22)
    global_ids = torch.empty(0, dtype=torch.long)
    counts = []
    base_reports = []
    appearance_reports = []
    paired_reports = []
    for index, guide_cpu in enumerate(guides):
        frontier = index * stride_frames
        commit_stop = frontier + stride_frames
        if guide_cpu.shape != (topology_spec.n_cells, topology_model.guide_dim):
            raise ValueError("guide raster does not match the topology model/grid")
        selected = generated_window_state(
            base_features,
            total_frames,
            frontier,
            stride_frames=stride_frames,
            support_sigma=support_sigma,
        )
        if len(base_features) != len(appearance_features):
            raise RuntimeError("paired streams lost stable row alignment")
        topology = predict_realizer_topology(
            topology_model,
            guide_cpu,
            selected.carried_global_features,
            total_frames=total_frames,
            frontier=frontier,
            stride_frames=stride_frames,
            support_sigma=support_sigma,
            topology_spec=topology_spec,
            realizer_spec=base_flow.grid_spec,
            occupancy_threshold=occupancy_threshold,
            device=target_device,
        )
        base_context = rasterize_scaffold_context(
            selected.context_features,
            context_standardizer,
            stride_frames=stride_frames,
            grid_spec=base_flow.grid_spec,
        ).to(target_device)
        if len(selected.context_row_indices):
            appearance_context_features = to_frontier_time(
                appearance_features[selected.context_row_indices],
                total_frames,
                frontier,
                stride_frames,
            )
        else:
            appearance_context_features = appearance_features.clone()
        appearance_context = rasterize_scaffold_context(
            appearance_context_features,
            context_standardizer,
            stride_frames=stride_frames,
            grid_spec=adapter.grid_spec,
        ).to(target_device)
        cells = topology.cell_indices.to(target_device)
        ranks = topology.slot_indices.to(target_device)
        if appearance_cell_weights is None:
            cell_weights = torch.ones(
                topology_spec.n_cells, device=target_device
            )
        else:
            cell_weights = appearance_cell_weights[index]
            if cell_weights.shape != (topology_spec.n_cells,):
                raise ValueError("appearance weights do not match the topology grid")
            cell_weights = cell_weights.to(target_device)
        sampled = sample_appearance_adapted_birth_marks(
            base_flow,
            adapter,
            base_context,
            appearance_context,
            cells,
            ranks,
            text_condition.to(target_device),
            cell_weights=cell_weights,
            steps=steps,
            generator=generator,
            guide_raster=guide_cpu.to(target_device),
        )
        base_unprojected = birth_standardizer.denormalize(sampled.base)
        appearance_unprojected = birth_standardizer.denormalize(sampled.appearance)
        allow_prefrontier = allow_initial_prefrontier and frontier == 0
        base_local = project_birth_topology(
            base_unprojected,
            cells,
            spec=base_flow.grid_spec,
            support_sigma=support_sigma,
            stride_frames=stride_frames,
            allow_prefrontier_support=allow_prefrontier,
        )
        appearance_local = base_local.clone()
        appearance_local[:, RGB_DIMENSIONS] = appearance_unprojected[
            :, RGB_DIMENSIONS
        ]
        local_error = _maximum_non_rgb_error(base_local, appearance_local)
        base_born = to_global_time(
            base_local.detach().cpu(), total_frames, frontier, stride_frames
        )
        converted_appearance = to_global_time(
            appearance_local.detach().cpu(), total_frames, frontier, stride_frames
        )
        appearance_born = base_born.clone()
        appearance_born[:, RGB_DIMENSIONS] = converted_appearance[:, RGB_DIMENSIONS]
        global_error = _maximum_non_rgb_error(base_born, appearance_born)
        if not torch.isfinite(base_born).all() or not torch.isfinite(
            appearance_born
        ).all():
            raise ValueError("mark flow or adapter produced non-finite jewel features")
        prior_base = base_features
        prior_appearance = appearance_features
        prior_ids = global_ids
        base_features = torch.cat((prior_base, base_born), dim=0)
        appearance_features = torch.cat((prior_appearance, appearance_born), dim=0)
        new_ids = torch.arange(len(prior_base), len(base_features), dtype=torch.long)
        global_ids = torch.cat((prior_ids, new_ids), dim=0)
        base_prior_error = (
            float((base_features[: len(prior_base)] - prior_base).abs().max())
            if len(prior_base)
            else 0.0
        )
        appearance_prior_error = (
            float(
                (
                    appearance_features[: len(prior_appearance)]
                    - prior_appearance
                )
                .abs()
                .max()
            )
            if len(prior_appearance)
            else 0.0
        )
        base_carried_error = (
            float(
                (
                    base_features[selected.carried_row_indices]
                    - selected.carried_global_features
                )
                .abs()
                .max()
            )
            if len(selected.carried_row_indices)
            else 0.0
        )
        appearance_carried_error = (
            float(
                (
                    appearance_features[selected.carried_row_indices]
                    - prior_appearance[selected.carried_row_indices]
                )
                .abs()
                .max()
            )
            if len(selected.carried_row_indices)
            else 0.0
        )
        if any(
            (
                base_prior_error,
                appearance_prior_error,
                base_carried_error,
                appearance_carried_error,
                local_error,
                global_error,
                sampled.max_non_rgb_error,
            )
        ):
            raise RuntimeError("appearance rollout violated immutable state ownership")
        maximum_count = int(topology.counts.max()) if len(topology.counts) else 0
        shared = {
            "index": index,
            "frontier": frontier,
            "commit_stop": commit_stop,
            "context_jewels": len(selected.context_row_indices),
            "carried_jewels": len(selected.carried_row_indices),
            "born_jewels": len(base_born),
            "state_jewels": len(base_features),
            "maximum_cell_count": maximum_count,
        }
        base_reports.append(
            ScaffoldMarkWindowReport(
                **shared,
                max_prior_feature_error=base_prior_error,
                max_carried_feature_error=base_carried_error,
            )
        )
        appearance_reports.append(
            ScaffoldMarkWindowReport(
                **shared,
                max_prior_feature_error=appearance_prior_error,
                max_carried_feature_error=appearance_carried_error,
            )
        )
        strengths = cell_weights[cells]
        paired_reports.append(
            AppearanceAdapterWindowReport(
                index=index,
                born_jewels=len(base_born),
                max_normalized_non_rgb_error=sampled.max_non_rgb_error,
                max_local_non_rgb_error=local_error,
                max_global_non_rgb_error=global_error,
                rgb_mae=_rgb_mae(base_born, appearance_born),
                mean_appearance_strength=(
                    float(strengths.mean()) if len(strengths) else 0.0
                ),
                active_appearance_fraction=(
                    float((strengths > 0).float().mean()) if len(strengths) else 0.0
                ),
            )
        )
        counts.append(topology.counts.clone())
    base = ScaffoldMarkRollout(
        base_features, global_ids.clone(), tuple(counts), tuple(base_reports)
    )
    appearance = ScaffoldMarkRollout(
        appearance_features,
        global_ids.clone(),
        tuple(count.clone() for count in counts),
        tuple(appearance_reports),
    )
    result = AppearanceAdapterRollout(base, appearance, tuple(paired_reports))
    if not (
        result.non_appearance_exact
        and result.lifecycle_exact
        and result.stable_ids_exact
        and result.topology_exact
    ):
        raise RuntimeError("appearance adapter escaped its topology/feature contract")
    return result
