"""Tests for prefix-conditioned sparse birth prediction."""

from __future__ import annotations

import unittest

import torch

from sol.streaming_data import BirthTarget
from sol.streaming_model import BirthContinuationModel
from sol.token_grid import GridSpec


class StreamingModelTests(unittest.TestCase):
    def test_training_and_decode_shapes(self) -> None:
        spec = GridSpec((2, 2, 2), 4)
        model = BirthContinuationModel(
            model_dim=32,
            grid_spec=spec,
            context_depth=1,
            cell_depth=1,
            slot_depth=1,
        )
        target = BirthTarget(
            values=torch.randn(3, 22),
            cell_indices=torch.tensor([0, 0, 5]),
            slot_indices=torch.tensor([0, 1, 0]),
            counts=torch.tensor([2, 0, 0, 0, 0, 1, 0, 0]),
            global_ids=torch.tensor([10, 11, 12]),
            birth_frames=torch.tensor([8, 8, 9]),
        )
        output = model.forward_training(torch.randn(8, 46), target)
        self.assertEqual(output.occupied_features.shape, (3, 22))
        self.assertEqual(output.log_count.shape, (8,))
        loss, terms = model.loss(output, target.values, target.counts)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(terms), {"feature", "count"})
        ablated = model.forward_from_context(torch.zeros(1, 32), target)
        self.assertEqual(ablated.occupied_features.shape, (3, 22))
        decoded = model.decode(torch.randn(8, 46))
        self.assertEqual(decoded.counts.shape, (8,))
        self.assertEqual(decoded.values.shape[1], 22)
        self.assertEqual(len(decoded.values), int(decoded.counts.sum()))

    def test_rejects_invalid_context_shape(self) -> None:
        model = BirthContinuationModel(
            model_dim=32, grid_spec=GridSpec((2, 2, 2), 4)
        )
        with self.assertRaises(ValueError):
            model.encode_context(torch.randn(7, 46))

    def test_local_context_preserves_one_token_per_cell(self) -> None:
        spec = GridSpec((2, 2, 2), 4)
        model = BirthContinuationModel(
            model_dim=32,
            grid_spec=spec,
            context_depth=1,
            cell_depth=1,
            slot_depth=1,
            context_mode="local",
        )
        context = model.encode_context(torch.randn(spec.n_cells, 46))
        self.assertEqual(context.shape, (1, spec.n_cells, 32))
        states = model.cell_states(context)
        self.assertEqual(states.shape, (1, spec.n_cells, 32))

    def test_rejects_invalid_context_mode(self) -> None:
        with self.assertRaises(ValueError):
            BirthContinuationModel(context_mode="flat")

    def test_text_condition_and_dropout_use_learned_null_path(self) -> None:
        spec = GridSpec((2, 2, 2), 4)
        model = BirthContinuationModel(
            model_dim=32,
            grid_spec=spec,
            context_depth=1,
            cell_depth=1,
            slot_depth=1,
            context_mode="local",
            text_dim=12,
        )
        target = BirthTarget(
            values=torch.randn(2, 22),
            cell_indices=torch.tensor([0, 3]),
            slot_indices=torch.tensor([0, 0]),
            counts=torch.tensor([1, 0, 0, 1, 0, 0, 0, 0]),
            global_ids=torch.tensor([4, 9]),
            birth_frames=torch.tensor([8, 9]),
        )
        context = model.encode_context(torch.randn(spec.n_cells, 46))
        text = torch.randn(1, 12)
        conditioned = model.forward_from_context(context, target, text)
        dropped = model.forward_from_context(
            context, target, text, torch.ones(1, dtype=torch.bool)
        )
        null = model.forward_from_context(context, target, None)
        self.assertFalse(
            torch.allclose(conditioned.occupied_features, null.occupied_features)
        )
        self.assertTrue(torch.allclose(dropped.occupied_features, null.occupied_features))
        self.assertTrue(torch.allclose(dropped.log_count, null.log_count))

    def test_text_condition_rejects_wrong_shape_and_unconfigured_model(self) -> None:
        spec = GridSpec((2, 2, 2), 4)
        context = torch.randn(1, 32)
        with self.assertRaises(ValueError):
            BirthContinuationModel(model_dim=32, grid_spec=spec).cell_states(
                context, torch.randn(1, 5)
            )
        prompted = BirthContinuationModel(
            model_dim=32, grid_spec=spec, text_dim=5
        )
        with self.assertRaises(ValueError):
            prompted.cell_states(context, torch.randn(2, 5))


if __name__ == "__main__":
    unittest.main()
