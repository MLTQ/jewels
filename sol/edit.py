"""Turn a cursor translation into protected jewels and local latent dirty cells."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.geometry import Parallelepiped, translate_selected
from sol.token_grid import GridSpec


@dataclass
class EditPlan:
    source: Parallelepiped
    destination: Parallelepiped
    delta: torch.Tensor
    selected_mask: torch.Tensor
    dirty_cells: torch.Tensor
    clean_context: torch.Tensor
    protected_moved: torch.Tensor

    def merge(self, generated_dirty: torch.Tensor) -> torch.Tensor:
        """Merge untouched context, locally regenerated jewels, and moved constraints."""
        feature_dim = self.clean_context.shape[-1]
        if generated_dirty.ndim != 2 or generated_dirty.shape[-1] != feature_dim:
            raise ValueError(f"generated_dirty must have shape (N,{feature_dim})")
        return torch.cat([self.clean_context, generated_dirty, self.protected_moved], dim=0)


def _dilate(mask: torch.Tensor, spec: GridSpec, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    gu, gv, gt = spec.shape
    volume = mask.reshape(gu, gv, gt)
    result = volume.clone()
    occupied = torch.nonzero(volume, as_tuple=False)
    for u, v, t in occupied.tolist():
        result[
            max(0, u - radius) : min(gu, u + radius + 1),
            max(0, v - radius) : min(gv, v + radius + 1),
            max(0, t - radius) : min(gt, t + radius + 1),
        ] = True
    return result.reshape(-1)


def plan_translation_edit(
    features: torch.Tensor,
    selection: Parallelepiped,
    delta: torch.Tensor,
    spec: GridSpec,
    *,
    halo_cells: int = 1,
) -> EditPlan:
    """Create a conservative source-to-destination latent inpainting plan.

    The union of endpoint AABBs also bounds the straight cursor sweep between
    them. A cell halo gives the latent model room to reconcile additive overlap
    and boundary appearance.
    """
    if halo_cells < 0:
        raise ValueError("halo_cells cannot be negative")
    edited, selected = translate_selected(features, selection, delta)
    destination = selection.translated(delta)
    src_min, src_max = selection.world_aabb()
    dst_min, dst_max = destination.world_aabb()
    swept_min = torch.minimum(src_min, dst_min)
    swept_max = torch.maximum(src_max, dst_max)
    dirty = spec.cells_for_aabb(
        swept_min, swept_max, device=features.device
    )
    dirty = _dilate(dirty, spec, halo_cells)

    edited_cells = spec.cell_index(edited[:, :3])
    clean = edited[~dirty[edited_cells] & ~selected]
    protected = edited[selected]
    return EditPlan(
        source=selection,
        destination=destination,
        delta=delta,
        selected_mask=selected,
        dirty_cells=dirty,
        clean_context=clean,
        protected_moved=protected,
    )
