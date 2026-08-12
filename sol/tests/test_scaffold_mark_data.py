"""Tests for initial-compatible scaffold mark corpus and generated state windows."""

from __future__ import annotations

import math
import unittest

import torch

from sol.scaffold_mark_data import (
    build_scaffold_mark_corpus,
    generated_window_state,
    rasterize_scaffold_context,
)
from sol.streaming_corpus import PromptedField
from sol.token_grid import GridSpec


def _features() -> torch.Tensor:
    features = torch.zeros(32, 22)
    features[:, 0] = torch.linspace(-0.9, 0.9, len(features))
    features[:, 1] = torch.linspace(0.9, -0.9, len(features))
    features[:, 2] = torch.linspace(-0.95, 0.95, len(features))
    features[:, 3] = 2 * math.log(0.08)
    features[:, 6] = 2 * math.log(0.08)
    features[:, 8] = 2 * math.log(0.14)
    features[:, 9:12] = 0.5
    features[:, 21] = 2.0
    return features


def _field(source_id: str, class_id: int, split: str) -> PromptedField:
    return PromptedField(
        source_id,
        class_id,
        f"class-{class_id}",
        split,
        _features(),
        32,
        (class_id,),
        (class_id + 2,),
    )


class ScaffoldMarkDataTests(unittest.TestCase):
    def test_corpus_includes_initial_marks_and_train_only_statistics(self) -> None:
        spec = GridSpec((4, 4, 2), 16)
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
        self.assertEqual(len(corpus.train), 2)
        self.assertEqual(len(corpus.validation), 2)
        self.assertEqual([view.frontier for view in corpus.train[0].views], [0, 8, 16, 24])
        initial = corpus.train[0].views[0]
        self.assertEqual(len(initial.context_features), 0)
        self.assertGreater(len(initial.births.values), 0)
        self.assertTrue(torch.isfinite(corpus.birth_standardizer.std).all())

        raster = rasterize_scaffold_context(
            initial.context_features,
            corpus.context_standardizer,
            stride_frames=8,
            grid_spec=spec,
        )
        self.assertEqual(raster.shape, (spec.n_cells, 46))
        self.assertEqual(float(raster.abs().max()), 0.0)

    def test_generated_state_reconstructs_training_window_selection(self) -> None:
        spec = GridSpec((4, 4, 2), 16)
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
        expected = corpus.train[0].views[1]
        selected = generated_window_state(
            _features(),
            32,
            8,
            stride_frames=8,
            support_sigma=2.0,
        )
        self.assertTrue(torch.equal(selected.context_row_indices, expected.context_ids))
        self.assertTrue(torch.equal(selected.carried_row_indices, expected.carried_ids))
        self.assertTrue(
            torch.equal(selected.carried_global_features, expected.carried_global_features)
        )


if __name__ == "__main__":
    unittest.main()
