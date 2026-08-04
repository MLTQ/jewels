"""Runnable CPU proof for the sol/ research contracts."""

from __future__ import annotations

import json

import torch

from sol.edit import plan_translation_edit
from sol.geometry import Parallelepiped
from sol.inpaint import masked_flow_inpaint
from sol.render import render_euclidean_knn, render_exact, render_truncated
from sol.synthetic import elongated_knn_counterexample, random_jewels
from sol.token_grid import GridSpec, OccupancyGrid


def _toy_velocity(
    state: torch.Tensor,
    _time: torch.Tensor,
    condition: torch.Tensor | None,
) -> torch.Tensor:
    bias = 0.0 if condition is None else condition.mean(dim=-1)[:, None, None]
    return -0.2 * state + bias


def main() -> None:
    torch.manual_seed(0)
    spec = GridSpec((8, 8, 4), slots_per_cell=256)
    grid = OccupancyGrid(spec)

    dense = random_jewels(45_000, seed=4)
    report = grid.capacity_report(dense)
    packed = grid.pack(dense)

    counterexample, query = elongated_knn_counterexample()
    exact = render_exact(counterexample, query)
    knn = render_euclidean_knn(counterexample, query, k=64)
    truncated = render_truncated(counterexample, query, support_sigma=5.0)

    selection = Parallelepiped.axis_aligned(
        torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.2, 0.2, 0.3])
    )
    delta = torch.tensor([0.45, 0.0, 0.0])
    edit = plan_translation_edit(dense[:4000], selection, delta, spec)

    known = torch.randn(1, spec.n_cells, 16)
    condition = torch.ones(1, 16)
    generator = torch.Generator().manual_seed(8)
    repaired = masked_flow_inpaint(
        _toy_velocity,
        known,
        edit.dirty_cells,
        condition=condition,
        cfg_scale=1.5,
        steps=8,
        generator=generator,
    )
    clean = ~edit.dirty_cells

    summary = {
        "dense_grid": {
            "jewels": dense.shape[0],
            "packed_jewels": int(packed.mask.sum()),
            "max_cell_occupancy": report.max_cell_occupancy,
            "slots_per_cell": spec.slots_per_cell,
            "fits_without_drop": report.fits,
        },
        "renderer": {
            "exact_rgb": exact[0].tolist(),
            "euclidean_knn_rgb": knn[0].tolist(),
            "conservative_rgb": truncated[0].tolist(),
            "knn_max_error": float((knn - exact).abs().max()),
            "conservative_max_error": float((truncated - exact).abs().max()),
        },
        "edit": {
            "selected_jewels": int(edit.selected_mask.sum()),
            "protected_moved_jewels": edit.protected_moved.shape[0],
            "dirty_cells": int(edit.dirty_cells.sum()),
            "clean_context_jewels": edit.clean_context.shape[0],
            "clean_latent_max_change": float(
                (repaired[:, clean] - known[:, clean]).abs().max()
            ),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
