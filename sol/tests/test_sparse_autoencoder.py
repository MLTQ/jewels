"""Sparse variable-count jewel decoder tests."""

from __future__ import annotations

import math
import unittest

import torch

from sol.sparse_autoencoder import (
    SparseAutoencoderOutput,
    SparseJewelAutoencoder,
    _cell_basis,
)
from sol.synthetic import random_jewels
from sol.token_grid import GridSpec


class SparseAutoencoderTests(unittest.TestCase):
    def _model(self) -> SparseJewelAutoencoder:
        return SparseJewelAutoencoder(
            feature_dim=22,
            model_dim=16,
            latent_dim=8,
            spec=GridSpec((2, 2, 1), slots_per_cell=32),
            enc_depth=1,
            dec_depth=1,
            heads=4,
            decode_chunk_size=7,
        )

    def test_training_materializes_only_occupied_jewels(self) -> None:
        model = self._model()
        features = random_jewels(40, seed=31)[None]
        target = model.grid.pack_compact(features)
        output = model.forward_compact(features, target)
        self.assertEqual(output.occupied_features.shape, features.shape)
        self.assertEqual(output.log_count.shape, (1, model.spec.n_cells))
        loss, terms = model.structural_loss(output, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(terms), {"feature", "count"})

    def test_neighboring_canonical_ranks_have_distinct_multiscale_basis(self) -> None:
        model = self._model()
        basis = model.decoder._slot_basis(torch.arange(16), torch.float32)
        self.assertGreater(float((basis[1] - basis[0]).norm()), 1.0)
        self.assertEqual(torch.unique(basis, dim=0).shape[0], 16)

    def test_local_only_encoder_preserves_raster_latent_shape(self) -> None:
        model = SparseJewelAutoencoder(
            feature_dim=22,
            model_dim=32,
            latent_dim=16,
            spec=GridSpec((4, 4, 2), slots_per_cell=16),
            enc_depth=0,
            dec_depth=1,
            heads=4,
        )
        features = random_jewels(40, seed=32)[None]
        self.assertEqual(model.encoder(features).shape, (1, 32, 16))

    def test_rank_encoder_is_invariant_to_input_permutation(self) -> None:
        model = SparseJewelAutoencoder(
            feature_dim=22,
            model_dim=32,
            latent_dim=16,
            spec=GridSpec((4, 4, 2), slots_per_cell=16),
            enc_depth=0,
            dec_depth=1,
            heads=4,
            encoder_mode="rank",
        )
        model.eval()
        features = random_jewels(40, seed=33)[None]
        permutation = torch.randperm(features.shape[1])
        with torch.no_grad():
            expected = model.encoder(features)
            actual = model.encoder(features[:, permutation])
        torch.testing.assert_close(actual, expected)

    def test_fourier_positions_remove_absolute_cell_tables(self) -> None:
        spec = GridSpec((4, 4, 2), slots_per_cell=16)
        model = SparseJewelAutoencoder(
            feature_dim=22,
            model_dim=32,
            latent_dim=16,
            spec=spec,
            enc_depth=0,
            dec_depth=1,
            heads=4,
            encoder_mode="rank",
            position_mode="fourier",
        )
        self.assertIsNone(model.encoder.cell_embed)
        self.assertIsNone(model.decoder.cell_embed)
        basis = _cell_basis(torch.arange(spec.n_cells), spec, torch.float32)
        self.assertEqual(basis.shape, (spec.n_cells, 27))
        self.assertEqual(torch.unique(basis, dim=0).shape[0], spec.n_cells)
        features = random_jewels(40, seed=34)[None]
        target = model.grid.pack_compact(features)
        output = model.forward_compact(features, target)
        loss, _ = model.structural_loss(output, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_fourier_positions_require_rank_encoder(self) -> None:
        with self.assertRaisesRegex(ValueError, "require encoder_mode='rank'"):
            SparseJewelAutoencoder(position_mode="fourier")

    def test_balanced_count_loss_upweights_rare_occupied_cells(self) -> None:
        model = SparseJewelAutoencoder(
            feature_dim=22,
            model_dim=16,
            latent_dim=8,
            spec=GridSpec((4, 4, 2), slots_per_cell=8),
            enc_depth=0,
            dec_depth=0,
            heads=4,
        )
        features = random_jewels(1, seed=35)[None]
        target = model.grid.pack_compact(features)
        output = SparseAutoencoderOutput(
            occupied_features=target.values.clone(),
            log_count=torch.zeros_like(target.counts, dtype=features.dtype),
        )
        _, global_terms = model.structural_loss(output, target)
        _, balanced_terms = model.structural_loss(
            output, target, balance_count=True
        )
        self.assertGreater(
            float(balanced_terms["count"]), float(global_terms["count"])
        )

    def test_inference_materializes_predicted_counts_without_padding(self) -> None:
        model = self._model()
        with torch.no_grad():
            model.decoder.count_head.weight.zero_()
            model.decoder.count_head.bias.fill_(math.log1p(3))
        latents = torch.randn(1, model.spec.n_cells, 8)
        decoded = model.decode(latents)[0]
        self.assertEqual(decoded.shape, (model.spec.n_cells * 3, 22))
        cells = model.spec.cell_index(decoded[:, :3])
        torch.testing.assert_close(
            torch.bincount(cells, minlength=model.spec.n_cells),
            torch.full((model.spec.n_cells,), 3),
        )


if __name__ == "__main__":
    unittest.main()
