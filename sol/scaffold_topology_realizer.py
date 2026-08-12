"""Bridge learned scaffold topology into the frozen stochastic mark realizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.birth_mark_flow import (
    BirthMarkFlowModel,
    project_birth_topology,
    sample_birth_marks,
)
from sol.scaffold_topology_data import rasterize_carried_state
from sol.scaffold_topology_eval import expand_topology_counts
from sol.streaming_data import FeatureStandardizer
from sol.token_grid import GridCapacityError, GridSpec


@dataclass(frozen=True)
class RealizerTopology:
    """One decoded cell-count field in the frozen realizer's rank convention."""

    counts: torch.Tensor
    cell_indices: torch.Tensor
    slot_indices: torch.Tensor


def validate_realizer_topology(
    counts: torch.Tensor,
    topology_spec: GridSpec,
    realizer_spec: GridSpec,
) -> RealizerTopology:
    """Validate grid/capacity compatibility and expand counts into nested ranks."""
    if topology_spec.shape != realizer_spec.shape:
        raise ValueError("topology and mark realizer use different grid shapes")
    if counts.shape != (topology_spec.n_cells,) or counts.dtype != torch.long:
        raise ValueError("decoded counts do not match the topology grid")
    maximum = int(counts.max()) if len(counts) else 0
    if maximum > realizer_spec.slots_per_cell:
        raise GridCapacityError(
            f"predicted topology needs {maximum} ranks but the frozen realizer "
            f"allows {realizer_spec.slots_per_cell}"
        )
    cell_indices, slot_indices = expand_topology_counts(
        counts, slots_per_cell=realizer_spec.slots_per_cell
    )
    return RealizerTopology(counts, cell_indices, slot_indices)


@torch.no_grad()
def predict_realizer_topology(
    model,
    guide_raster: torch.Tensor,
    carried_global_features: torch.Tensor,
    *,
    total_frames: int,
    frontier: int,
    stride_frames: int,
    support_sigma: float,
    topology_spec: GridSpec,
    realizer_spec: GridSpec,
    occupancy_threshold: float,
    device: str | torch.device,
) -> RealizerTopology:
    """Predict counts from scaffold/carry and decode them for a frozen mark model."""
    target_device = torch.device(device)
    carry_raster = rasterize_carried_state(
        carried_global_features.cpu(),
        total_frames,
        frontier,
        stride_frames,
        topology_spec,
        support_sigma=support_sigma,
    )
    model.eval()
    output = model(guide_raster.to(target_device), carry_raster.to(target_device))
    counts = model.decode_counts(
        output, occupancy_threshold=occupancy_threshold
    ).long().cpu()
    return validate_realizer_topology(counts, topology_spec, realizer_spec)


@torch.no_grad()
def realize_topology_marks(
    flow: BirthMarkFlowModel,
    context_raster: torch.Tensor,
    topology: RealizerTopology,
    text_condition: torch.Tensor,
    birth_standardizer: FeatureStandardizer,
    *,
    guide_raster: torch.Tensor,
    support_sigma: float,
    stride_frames: int,
    steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample and hard-project frontier-local marks for learned topology."""
    device = context_raster.device
    cells = topology.cell_indices.to(device)
    ranks = topology.slot_indices.to(device)
    normalized = sample_birth_marks(
        flow,
        context_raster,
        cells,
        ranks,
        text_condition.to(device),
        steps=steps,
        generator=generator,
        guide_raster=guide_raster.to(device),
    )
    local = birth_standardizer.denormalize(normalized)
    return project_birth_topology(
        local,
        cells,
        spec=flow.grid_spec,
        support_sigma=support_sigma,
        stride_frames=stride_frames,
    )
