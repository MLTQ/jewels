"""Tests for learned hierarchical phrase decoder data and model contracts."""

from __future__ import annotations

import unittest

import torch

from sol.factorized_jewel_casting_language import fit_factorized_codebook
from sol.hierarchical_jewel_decoder import (
    HierarchicalPhraseDecoder,
    build_hierarchical_phrase_batch,
    build_sampled_hierarchical_phrase_batch,
    phrase_decoder_loss,
    phrase_values_to_features,
    residual_scale,
)
from sol.synthetic import random_jewels
from sol.token_grid import GridSpec


class HierarchicalJewelDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = GridSpec((2, 2, 2), slots_per_cell=1)
        fields = [random_jewels(96, seed=seed) for seed in range(3)]
        self.target = fields[2]
        self.pair_codebook, _ = fit_factorized_codebook(
            fields[:2], spec=self.spec, bundle_size=2,
            vocabulary_size=8, iterations=2, assignment_chunk=16,
        )
        self.individual_codebook, _ = fit_factorized_codebook(
            fields[:2], spec=self.spec, bundle_size=1,
            vocabulary_size=8, iterations=2, assignment_chunk=16,
        )
        self.batch, self.pair, _ = build_hierarchical_phrase_batch(
            self.target, self.pair_codebook, self.individual_codebook
        )

    def test_target_values_decode_exactly(self) -> None:
        decoded = phrase_values_to_features(
            self.batch.target_values, self.pair, self.pair_codebook
        )
        for dimension in range(22):
            torch.testing.assert_close(
                decoded[:, dimension].sort().values,
                self.target[:, dimension].sort().values,
                rtol=1e-5, atol=1e-5,
            )

    def test_decoder_uses_phrase_shape_and_masked_loss(self) -> None:
        scale = residual_scale(self.batch)
        model = HierarchicalPhraseDecoder(
            vocabulary_size=8, n_cells=self.spec.n_cells,
            embedding_dim=8, hidden_dim=32, depth=2, output_scale=scale,
        )
        prediction = model(self.batch)
        self.assertEqual(prediction.shape, self.batch.target_values.shape)
        self.assertTrue(torch.isfinite(phrase_decoder_loss(prediction, self.batch, scale)))

    def test_padding_token_marks_only_missing_second_jewels(self) -> None:
        padding = self.pair_codebook.vocabulary_size
        missing = self.batch.counts == 1
        self.assertTrue(torch.equal(self.batch.tokens[:, 3] == padding, missing))
        self.assertTrue(torch.equal(self.batch.tokens[:, 5] == padding, missing))

    def test_sampled_builder_matches_full_builder_when_all_pairs_selected(self) -> None:
        # The order is random, so compare the exact target rows after sorting by anchors.
        sampled = build_sampled_hierarchical_phrase_batch(
            self.target,
            self.pair_codebook,
            self.individual_codebook,
            max_pairs=len(self.batch),
            generator=torch.Generator().manual_seed(7),
        )
        full_order = torch.argsort(self.batch.anchors[:, 0])
        sampled_order = torch.argsort(sampled.anchors[:, 0])
        torch.testing.assert_close(
            sampled.target_values[sampled_order], self.batch.target_values[full_order]
        )
        torch.testing.assert_close(
            sampled.base_values[sampled_order], self.batch.base_values[full_order]
        )


if __name__ == "__main__":
    unittest.main()
