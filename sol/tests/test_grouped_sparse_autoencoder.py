"""Occupied-group jewel tokenizer tests."""

from __future__ import annotations

import unittest

import torch

from sol.grouped_sparse_autoencoder import GroupedSparseJewelAutoencoder
from sol.synthetic import random_jewels
from sol.token_grid import GridSpec


class GroupedSparseAutoencoderTests(unittest.TestCase):
    def _model(self) -> GroupedSparseJewelAutoencoder:
        return GroupedSparseJewelAutoencoder(
            feature_dim=22,
            model_dim=32,
            latent_dim=12,
            spec=GridSpec((4, 4, 2), slots_per_cell=32),
            jewels_per_token=4,
            enc_depth=1,
            dec_depth=1,
            decode_chunk_size=7,
        )

    def test_topology_is_compact_and_count_exact(self) -> None:
        model = self._model()
        features = random_jewels(40, seed=71)[None]
        latents = model.encoder(features)
        self.assertEqual(int(latents.group_counts.sum()), 40)
        self.assertTrue((latents.group_counts <= 4).all())
        self.assertLessEqual(len(latents.values), 40)
        decoded = model.decode(latents)[0]
        self.assertEqual(decoded.shape, (40, 22))
        expected_cells = model.spec.cell_index(features[0, :, :3])
        decoded_cells = model.spec.cell_index(decoded[:, :3])
        torch.testing.assert_close(
            torch.bincount(decoded_cells, minlength=model.spec.n_cells),
            torch.bincount(expected_cells, minlength=model.spec.n_cells),
        )

    def test_encoder_is_invariant_to_input_permutation(self) -> None:
        model = self._model().eval()
        features = random_jewels(40, seed=72)[None]
        permutation = torch.randperm(features.shape[1])
        with torch.no_grad():
            expected = model.encoder(features)
            actual = model.encoder(features[:, permutation])
        torch.testing.assert_close(actual.values, expected.values)
        torch.testing.assert_close(actual.cell_indices, expected.cell_indices)
        torch.testing.assert_close(actual.group_indices, expected.group_indices)
        torch.testing.assert_close(actual.group_counts, expected.group_counts)

    def test_compact_training_loss_backpropagates(self) -> None:
        model = self._model()
        features = random_jewels(40, seed=73)[None]
        target = model.grid.pack_compact(features)
        output = model.forward_compact(features, target)
        self.assertEqual(output.occupied_features.shape, features.shape)
        loss, terms = model.structural_loss(output, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(terms), {"feature", "count"})
        self.assertEqual(float(terms["count"]), 0.0)


if __name__ == "__main__":
    unittest.main()
