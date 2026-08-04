"""Geometry, edit-plan, and latent-clamping tests."""

from __future__ import annotations

import unittest

import torch

from sol.edit import plan_translation_edit
from sol.geometry import Parallelepiped, translate_selected
from sol.inpaint import masked_flow_inpaint
from sol.synthetic import random_jewels
from sol.token_grid import GridSpec


def _unit_velocity(
    state: torch.Tensor,
    _time: torch.Tensor,
    condition: torch.Tensor | None,
) -> torch.Tensor:
    scale = 1.0 if condition is None else condition[:, :1, None]
    return torch.ones_like(state) * scale


class _MaskAwareVelocity:
    mask_conditioning = True

    def __init__(self) -> None:
        self.seen_mask: torch.Tensor | None = None

    def __call__(
        self,
        state: torch.Tensor,
        _time: torch.Tensor,
        _condition: torch.Tensor | None,
        *,
        edit_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.seen_mask = edit_mask
        return torch.zeros_like(state)


class EditAndInpaintTests(unittest.TestCase):
    def test_translation_changes_selected_centers_only(self) -> None:
        features = random_jewels(100, seed=5)
        box = Parallelepiped.axis_aligned(torch.zeros(3), torch.full((3,), 0.5))
        delta = torch.tensor([0.2, -0.1, 0.05])
        edited, selected = translate_selected(features, box, delta)
        torch.testing.assert_close(
            edited[selected, :3], features[selected, :3] + delta
        )
        torch.testing.assert_close(edited[~selected], features[~selected])
        torch.testing.assert_close(edited[selected, 3:], features[selected, 3:])

    def test_edit_plan_marks_sweep_and_protects_moved_jewels(self) -> None:
        features = random_jewels(500, seed=6)
        spec = GridSpec((8, 8, 4), slots_per_cell=64)
        box = Parallelepiped.axis_aligned(
            torch.zeros(3), torch.tensor([0.2, 0.2, 0.3])
        )
        delta = torch.tensor([0.5, 0.0, 0.0])
        plan = plan_translation_edit(features, box, delta, spec, halo_cells=1)
        self.assertEqual(plan.protected_moved.shape[0], int(plan.selected_mask.sum()))
        self.assertGreater(int(plan.dirty_cells.sum()), 0)
        clean_cells = spec.cell_index(plan.clean_context[:, :3])
        self.assertFalse(bool(plan.dirty_cells[clean_cells].any()))

    def test_inpainting_clamps_every_clean_cell_exactly(self) -> None:
        known = torch.randn(2, 12, 6)
        dirty = torch.zeros(12, dtype=torch.bool)
        dirty[[2, 3, 8]] = True
        condition = torch.tensor([[2.0], [3.0]])
        generator = torch.Generator().manual_seed(12)
        result = masked_flow_inpaint(
            _unit_velocity,
            known,
            dirty,
            condition=condition,
            cfg_scale=1.0,
            steps=5,
            generator=generator,
        )
        torch.testing.assert_close(result[:, ~dirty], known[:, ~dirty], rtol=0, atol=0)
        self.assertGreater(float((result[:, dirty] - known[:, dirty]).abs().max()), 0)

    def test_inpainting_passes_dirty_mask_to_mask_aware_model(self) -> None:
        model = _MaskAwareVelocity()
        dirty = torch.tensor([False, True, False, True])
        masked_flow_inpaint(
            model,
            torch.randn(1, 4, 3),
            dirty,
            steps=2,
            generator=torch.Generator().manual_seed(3),
        )
        self.assertIsNotNone(model.seen_mask)
        torch.testing.assert_close(model.seen_mask[0], dirty)


if __name__ == "__main__":
    unittest.main()
