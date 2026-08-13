"""Tests for the compact scaffold-gated RGB adapter."""

from __future__ import annotations

import unittest

import torch

from sol.birth_mark_flow import BirthMarkFlowModel, sample_birth_marks
from sol.scaffold_appearance_adapter import (
    NON_RGB_DIMENSIONS,
    RGB_DIMENSIONS,
    ScaffoldAppearanceAdapter,
    sample_appearance_adapted_birth_marks,
    top_fraction_cell_gate,
)
from sol.token_grid import GridSpec


class ScaffoldAppearanceAdapterTests(unittest.TestCase):
    def _models(
        self, spec: GridSpec
    ) -> tuple[BirthMarkFlowModel, ScaffoldAppearanceAdapter]:
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
        adapter = ScaffoldAppearanceAdapter(
            model_dim=8,
            depth=1,
            grid_spec=spec,
            text_dim=8,
        )
        return base, adapter

    def test_zero_start_and_compact_default(self) -> None:
        default = ScaffoldAppearanceAdapter()
        self.assertLess(sum(parameter.numel() for parameter in default.parameters()), 100_000)
        spec = GridSpec((2, 2, 2), 4)
        _, adapter = self._models(spec)
        output = adapter(
            torch.randn(spec.n_cells, 46),
            torch.randn(4, 22),
            torch.randn(4, 22),
            torch.tensor([0.4]),
            torch.tensor([0, 0, 3, 7]),
            torch.tensor([0, 1, 0, 0]),
            torch.randn(8),
            guide_raster=torch.randn(spec.n_cells, 3),
        )
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_base_is_exact_and_adapter_changes_only_gated_rgb(self) -> None:
        torch.manual_seed(4)
        spec = GridSpec((2, 2, 2), 4)
        base, adapter = self._models(spec)
        with torch.no_grad():
            adapter.rgb_velocity_head.bias.fill_(0.5)
        context = torch.randn(spec.n_cells, 46)
        guide = torch.randn(spec.n_cells, 3)
        cells = torch.tensor([0, 1, 3, 7])
        ranks = torch.tensor([0, 0, 0, 0])
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
        weights = torch.tensor([1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0])
        sampled = sample_appearance_adapted_birth_marks(
            base,
            adapter,
            context,
            context,
            cells,
            ranks,
            text,
            cell_weights=weights,
            steps=3,
            generator=torch.Generator().manual_seed(9),
            guide_raster=guide,
        )
        self.assertTrue(torch.equal(sampled.base, expected))
        self.assertEqual(sampled.max_non_rgb_error, 0.0)
        self.assertTrue(
            torch.equal(
                sampled.base[:, NON_RGB_DIMENSIONS],
                sampled.appearance[:, NON_RGB_DIMENSIONS],
            )
        )
        self.assertTrue(torch.equal(sampled.base[1], sampled.appearance[1]))
        self.assertTrue(torch.equal(sampled.base[3], sampled.appearance[3]))
        self.assertGreater(
            float(
                (
                    sampled.base[[0, 2]][:, RGB_DIMENSIONS]
                    - sampled.appearance[[0, 2]][:, RGB_DIMENSIONS]
                )
                .abs()
                .max()
            ),
            0.0,
        )

    def test_top_fraction_gate_has_exact_cardinality(self) -> None:
        scores = torch.arange(10, dtype=torch.float32)
        gate = top_fraction_cell_gate(scores, 0.2)
        self.assertEqual(int(gate.sum()), 2)
        self.assertTrue(torch.equal(gate.nonzero()[:, 0], torch.tensor([8, 9])))


if __name__ == "__main__":
    unittest.main()
