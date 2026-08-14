"""Autonomous initial-plus-continuation jewel generation from video scaffolds."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch

from sol.birth_mark_flow import BirthMarkFlowModel
from sol.scaffold_mark_data import generated_window_state, rasterize_scaffold_context
from sol.scaffold_topology import ScaffoldTopologyModel
from sol.scaffold_topology_realizer import (
    predict_realizer_topology,
    realize_topology_marks,
    validate_realizer_topology,
)
from sol.streaming_data import FeatureStandardizer
from sol.streaming_features import to_global_time
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class ScaffoldMarkWindowReport:
    """State, capacity, and immutability audit for one generated stride."""

    index: int
    frontier: int
    commit_stop: int
    context_jewels: int
    carried_jewels: int
    born_jewels: int
    state_jewels: int
    maximum_cell_count: int
    max_prior_feature_error: float
    max_carried_feature_error: float

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "frontier": self.frontier,
            "commit_stop": self.commit_stop,
            "context_jewels": self.context_jewels,
            "carried_jewels": self.carried_jewels,
            "born_jewels": self.born_jewels,
            "state_jewels": self.state_jewels,
            "maximum_cell_count": self.maximum_cell_count,
            "max_prior_feature_error": self.max_prior_feature_error,
            "max_carried_feature_error": self.max_carried_feature_error,
        }


@dataclass(frozen=True)
class ScaffoldMarkRollout:
    """Append-only generated field and the decoded topology for every stride."""

    features: torch.Tensor
    global_ids: torch.Tensor
    counts: tuple[torch.Tensor, ...]
    windows: tuple[ScaffoldMarkWindowReport, ...]
    topology_contract: str = "paired_rollout_managed_topology"

    @property
    def completed_frames(self) -> int:
        return self.windows[-1].commit_stop if self.windows else 0

    @property
    def report(self) -> dict:
        return {
            "completed_frames": self.completed_frames,
            "total_jewels": len(self.features),
            "stable_ids_exact": torch.equal(
                self.global_ids, torch.arange(len(self.global_ids))
            ),
            "topology_contract": self.topology_contract,
            "windows": [window.to_dict() for window in self.windows],
        }


@torch.no_grad()
def rollout_scaffold_marks(
    topology_model: ScaffoldTopologyModel,
    mark_flow: BirthMarkFlowModel,
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
    owned_counts: Sequence[torch.Tensor] | None = None,
) -> ScaffoldMarkRollout:
    """Generate frontier zero and every later stride from model-produced state."""
    if not guides or len(guides) * stride_frames > total_frames:
        raise ValueError("guides must describe one or more complete strides")
    if owned_counts is not None and len(owned_counts) != len(guides):
        raise ValueError("owned topology must provide one count raster per guide")
    if topology_spec.shape != mark_flow.grid_spec.shape:
        raise ValueError("topology and mark flow use different grid shapes")
    if mark_flow.grid_spec.slots_per_cell < topology_spec.slots_per_cell:
        raise ValueError("mark flow cannot realize the topology rank capacity")
    target_device = torch.device(device)
    topology_model.eval()
    mark_flow.eval()
    features = torch.empty(0, 22)
    global_ids = torch.empty(0, dtype=torch.long)
    counts = []
    reports = []
    for index, guide_cpu in enumerate(guides):
        frontier = index * stride_frames
        commit_stop = frontier + stride_frames
        if guide_cpu.shape != (topology_spec.n_cells, topology_model.guide_dim):
            raise ValueError("guide raster does not match the topology model/grid")
        selected = generated_window_state(
            features,
            total_frames,
            frontier,
            stride_frames=stride_frames,
            support_sigma=support_sigma,
        )
        if owned_counts is None:
            topology = predict_realizer_topology(
                topology_model,
                guide_cpu,
                selected.carried_global_features,
                total_frames=total_frames,
                frontier=frontier,
                stride_frames=stride_frames,
                support_sigma=support_sigma,
                topology_spec=topology_spec,
                realizer_spec=mark_flow.grid_spec,
                occupancy_threshold=occupancy_threshold,
                device=target_device,
            )
        else:
            topology = validate_realizer_topology(
                owned_counts[index].detach().cpu(),
                topology_spec,
                mark_flow.grid_spec,
            )
        context = rasterize_scaffold_context(
            selected.context_features,
            context_standardizer,
            stride_frames=stride_frames,
            grid_spec=mark_flow.grid_spec,
        ).to(target_device)
        local = realize_topology_marks(
            mark_flow,
            context,
            topology,
            text_condition,
            birth_standardizer,
            guide_raster=guide_cpu.to(target_device),
            support_sigma=support_sigma,
            stride_frames=stride_frames,
            steps=steps,
            generator=generator,
            allow_prefrontier_support=allow_initial_prefrontier and frontier == 0,
        )
        born = to_global_time(
            local.detach().cpu(), total_frames, frontier, stride_frames
        )
        if not torch.isfinite(born).all():
            raise ValueError("mark flow produced non-finite jewel features")
        prior = features
        prior_ids = global_ids
        features = torch.cat((prior, born), dim=0)
        new_ids = torch.arange(len(prior), len(features), dtype=torch.long)
        global_ids = torch.cat((prior_ids, new_ids), dim=0)
        prior_error = (
            float((features[: len(prior)] - prior).abs().max()) if len(prior) else 0.0
        )
        carried_error = (
            float(
                (
                    features[selected.carried_row_indices]
                    - selected.carried_global_features
                )
                .abs()
                .max()
            )
            if len(selected.carried_row_indices)
            else 0.0
        )
        if prior_error or carried_error:
            raise RuntimeError("append-only generation modified persistent jewel features")
        counts.append(topology.counts.clone())
        reports.append(
            ScaffoldMarkWindowReport(
                index=index,
                frontier=frontier,
                commit_stop=commit_stop,
                context_jewels=len(selected.context_row_indices),
                carried_jewels=len(selected.carried_row_indices),
                born_jewels=len(born),
                state_jewels=len(features),
                maximum_cell_count=(
                    int(topology.counts.max()) if len(topology.counts) else 0
                ),
                max_prior_feature_error=prior_error,
                max_carried_feature_error=carried_error,
            )
        )
    if not torch.equal(global_ids, torch.arange(len(global_ids))):
        raise RuntimeError("generated stable IDs are not append-only and contiguous")
    return ScaffoldMarkRollout(
        features,
        global_ids,
        tuple(counts),
        tuple(reports),
        (
            "self_predicted_from_generated_carry"
            if owned_counts is None
            else "externally_owned_cell_counts"
        ),
    )
