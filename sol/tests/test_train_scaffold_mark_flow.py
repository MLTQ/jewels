"""Tests for preparing initial and continuation scaffold mark training rows."""

from __future__ import annotations

import unittest

import torch

from sol.scaffold_mark_data import build_scaffold_mark_corpus
from sol.streaming_corpus import PromptedField
from sol.token_grid import GridSpec
from sol.train_scaffold_mark_flow import _feature_objective, _prepare


def _field(source_id: str, class_id: int, split: str) -> PromptedField:
    features = torch.zeros(16, 22)
    features[:, :2] = torch.rand(16, 2) * 1.6 - 0.8
    features[:, 2] = torch.linspace(-0.9, 0.9, 16)
    features[:, 3] = -5.0
    features[:, 6] = -5.0
    features[:, 8] = -3.0
    features[:, 9:12] = 0.5
    features[:, 21] = 2.0
    return PromptedField(
        source_id,
        class_id,
        f"class-{class_id}",
        split,
        features,
        16,
        (class_id,),
        (class_id + 2,),
    )


class TrainScaffoldMarkFlowTests(unittest.TestCase):
    def test_feature_objective_emphasizes_salient_cells(self) -> None:
        expected = torch.zeros(2, 3)
        salient_error = expected.clone()
        salient_error[1] = 1
        quiet_error = expected.clone()
        quiet_error[0] = 1
        cells = torch.tensor([0, 1])
        saliency = torch.tensor([0.1, 2.0])
        salient_loss = _feature_objective(
            salient_error, expected, cells, saliency, 2.0
        )
        quiet_loss = _feature_objective(
            quiet_error, expected, cells, saliency, 2.0
        )
        self.assertGreater(float(salient_loss), float(quiet_loss))

    def test_spatial_saliency_keeps_temporal_dimensions_uniform(self) -> None:
        expected = torch.zeros(2, 3)
        high_cell_error = expected.clone()
        high_cell_error[1, 2] = 1
        low_cell_error = expected.clone()
        low_cell_error[0, 2] = 1
        cells = torch.tensor([0, 1])
        saliency = torch.tensor([0.1, 2.0])
        high_loss = _feature_objective(
            high_cell_error, expected, cells, saliency, 2.0, (0, 1)
        )
        low_loss = _feature_objective(
            low_cell_error, expected, cells, saliency, 2.0, (0, 1)
        )
        self.assertAlmostEqual(float(high_loss), float(low_loss), places=6)

    def test_prepare_retains_zero_context_initial_rows(self) -> None:
        torch.manual_seed(7)
        spec = GridSpec((2, 2, 2), 8)
        corpus = build_scaffold_mark_corpus(
            [
                _field("train-a", 0, "train"),
                _field("train-b", 1, "train"),
                _field("valid-a", 0, "validation"),
                _field("valid-b", 1, "validation"),
            ],
            torch.randn(4, 8),
            stride_frames=8,
            support_sigma=2.0,
            grid_spec=spec,
        )
        guides = {
            (source.field.source_id, view.index): torch.rand(spec.n_cells, 3)
            for source in corpus.sources
            for view in source.views
        }
        prepared = _prepare(corpus, guides, torch.device("cpu"))
        self.assertEqual(len(prepared), 4)
        initial = [view for view in prepared if view.frontier == 0]
        self.assertEqual(len(initial), 2)
        self.assertTrue(all(float(view.context_raster.abs().max()) == 0 for view in initial))
        self.assertTrue(all(view.slot_indices.max() < spec.slots_per_cell for view in prepared))
        self.assertTrue(all(view.background.shape == (3,) for view in prepared))
        self.assertTrue(
            all(view.cell_saliency.shape == (spec.n_cells,) for view in prepared)
        )
        self.assertTrue(all((view.cell_saliency > 0).all() for view in prepared))
        self.assertTrue(all(view.total_frames == 16 for view in prepared))


if __name__ == "__main__":
    unittest.main()
