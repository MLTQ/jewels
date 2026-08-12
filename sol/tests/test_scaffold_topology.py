"""Tests for scaffold-conditioned topology prediction."""

from __future__ import annotations

import unittest

import torch

from sol.scaffold_topology import ScaffoldTopologyModel
from sol.token_grid import GridSpec


class ScaffoldTopologyTests(unittest.TestCase):
    def test_forward_loss_and_decode_contracts(self) -> None:
        spec = GridSpec((2, 2, 2), 8)
        model = ScaffoldTopologyModel(
            model_dim=32,
            grid_spec=spec,
            encoder_depth=1,
            cell_depth=1,
        )
        guide = torch.rand(2, spec.n_cells, 3)
        carry = torch.rand(2, spec.n_cells, 3)
        target = torch.tensor(
            [[2, 0, 1, 3, 0, 1, 2, 1], [0, 1, 2, 0, 3, 2, 1, 0]]
        )
        output = model(guide, carry)
        self.assertEqual(output.occupancy_logits.shape, target.shape)
        self.assertTrue((output.positive_counts >= 1).all())
        loss, terms = model.loss(output, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(terms),
            {"occupancy", "positive_count", "total_count", "distribution"},
        )
        decoded = model.decode_counts(output)
        self.assertEqual(decoded.shape, target.shape)
        self.assertTrue((decoded >= 0).all())
        self.assertTrue((decoded <= spec.slots_per_cell).all())

    def test_missing_carry_defaults_to_zero_and_bad_shapes_fail(self) -> None:
        spec = GridSpec((2, 2, 2), 4)
        model = ScaffoldTopologyModel(
            model_dim=32,
            grid_spec=spec,
            encoder_depth=1,
            cell_depth=1,
        )
        guide = torch.rand(spec.n_cells, 3)
        implicit = model(guide)
        explicit = model(guide, torch.zeros(spec.n_cells, 3))
        self.assertTrue(
            torch.allclose(implicit.occupancy_logits, explicit.occupancy_logits)
        )
        with self.assertRaises(ValueError):
            model(torch.rand(spec.n_cells - 1, 3))
        with self.assertRaises(ValueError):
            model.decode_counts(implicit, occupancy_threshold=1.0)

    def test_unbatched_loss_is_supported(self) -> None:
        spec = GridSpec((2, 2, 2), 4)
        model = ScaffoldTopologyModel(
            model_dim=32,
            grid_spec=spec,
            encoder_depth=1,
            cell_depth=1,
        )
        target = torch.tensor([1, 2, 0, 1, 0, 3, 2, 0])
        output = model(torch.rand(spec.n_cells, 3))
        loss, _ = model.loss(output, target)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
