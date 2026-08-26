"""Tests for hierarchical decoder evaluation helpers."""

from __future__ import annotations

import unittest

import torch

from sol.hierarchical_jewel_decoder import (
    HierarchicalPhraseBatch,
    HierarchicalPhraseDecoder,
)
from sol.train_hierarchical_jewel_decoder import dataset_loss, predict_values


class HierarchicalDecoderTrainingTests(unittest.TestCase):
    def test_chunked_prediction_and_loss_cover_terminal_chunk(self) -> None:
        count = 11
        batch = HierarchicalPhraseBatch(
            tokens=torch.zeros(count, 6, dtype=torch.long),
            cells=torch.zeros(count, dtype=torch.long),
            anchors=torch.zeros(count, 3),
            counts=torch.full((count,), 2, dtype=torch.long),
            base_values=torch.zeros(count, 2, 22),
            target_values=torch.zeros(count, 2, 22),
        )
        model = HierarchicalPhraseDecoder(
            vocabulary_size=8, n_cells=2, embedding_dim=8,
            hidden_dim=16, depth=1, output_scale=torch.ones(2, 22),
        )
        prediction = predict_values(model, batch, chunk=4)
        self.assertEqual(prediction.shape, batch.target_values.shape)
        self.assertTrue(torch.isfinite(torch.tensor(dataset_loss(model, batch, torch.ones(2, 22), chunk=4))))


if __name__ == "__main__":
    unittest.main()
