"""Tests for lifecycle-locked paired mark-flow sampling."""

from __future__ import annotations

import copy
import unittest

import torch

from sol.birth_mark_flow import BirthMarkFlowModel, sample_birth_marks
from sol.lifecycle_appearance_flow import (
    APPEARANCE_DIMENSION_SETS,
    LIFECYCLE_DIMENSIONS,
    SPATIAL_APPEARANCE_DIMENSIONS,
    sample_lifecycle_locked_birth_marks,
)
from sol.token_grid import GridSpec


class LifecycleAppearanceFlowTests(unittest.TestCase):
    def test_base_reproduces_ordinary_sampling_and_candidate_lifecycle_is_exact(self) -> None:
        torch.manual_seed(4)
        spec = GridSpec((2, 2, 2), 4)
        base = BirthMarkFlowModel(
            model_dim=8,
            grid_spec=spec,
            context_depth=1,
            noisy_depth=1,
            guide_depth=1,
            cell_depth=1,
            mark_depth=1,
            text_dim=8,
            guide_dim=3,
            guide_heads=1,
        )
        appearance = copy.deepcopy(base)
        with torch.no_grad():
            appearance.velocity_head.bias.fill_(1.0)
        context = torch.randn(spec.n_cells, 46)
        guide = torch.randn(spec.n_cells, 3)
        cells = torch.tensor([0, 0, 3, 7])
        ranks = torch.tensor([0, 1, 0, 0])
        text = torch.randn(8)
        expected = sample_birth_marks(
            base,
            context,
            cells,
            ranks,
            text,
            steps=3,
            generator=torch.Generator().manual_seed(9),
            guide_raster=guide,
        )
        sampled = sample_lifecycle_locked_birth_marks(
            base,
            appearance,
            context,
            context,
            cells,
            ranks,
            text,
            steps=3,
            generator=torch.Generator().manual_seed(9),
            guide_raster=guide,
        )
        self.assertTrue(torch.equal(sampled.base, expected))
        self.assertTrue(sampled.lifecycle_exact)
        self.assertTrue(
            torch.equal(
                sampled.base[:, LIFECYCLE_DIMENSIONS],
                sampled.appearance[:, LIFECYCLE_DIMENSIONS],
            )
        )
        self.assertGreater(
            float(
                (
                    sampled.base[:, SPATIAL_APPEARANCE_DIMENSIONS]
                    - sampled.appearance[:, SPATIAL_APPEARANCE_DIMENSIONS]
                )
                .abs()
                .max()
            ),
            0.0,
        )

        static = sample_lifecycle_locked_birth_marks(
            base,
            appearance,
            context,
            context,
            cells,
            ranks,
            text,
            steps=3,
            generator=torch.Generator().manual_seed(9),
            guide_raster=guide,
            appearance_dimensions=APPEARANCE_DIMENSION_SETS["static-detail"],
            appearance_strengths=torch.tensor([1.0, 0.0, 0.5, 1.0]),
        )
        frozen = tuple(
            index
            for index in range(22)
            if index not in APPEARANCE_DIMENSION_SETS["static-detail"]
        )
        self.assertTrue(torch.equal(static.base[:, frozen], static.appearance[:, frozen]))
        self.assertTrue(torch.equal(static.base[1], static.appearance[1]))

    def test_incompatible_grid_is_rejected(self) -> None:
        base = BirthMarkFlowModel(
            model_dim=8,
            grid_spec=GridSpec((1, 1, 1), 2),
            context_depth=1,
            noisy_depth=1,
            cell_depth=1,
            mark_depth=1,
            text_dim=8,
            guide_heads=1,
        )
        appearance = BirthMarkFlowModel(
            model_dim=8,
            grid_spec=GridSpec((1, 1, 1), 3),
            context_depth=1,
            noisy_depth=1,
            cell_depth=1,
            mark_depth=1,
            text_dim=8,
            guide_heads=1,
        )
        with self.assertRaisesRegex(ValueError, "different topology grids"):
            sample_lifecycle_locked_birth_marks(
                base,
                appearance,
                torch.zeros(1, 46),
                torch.zeros(1, 46),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(8),
            )


if __name__ == "__main__":
    unittest.main()
