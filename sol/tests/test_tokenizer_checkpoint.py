"""Tokenizer checkpoint architecture dispatch tests."""

from __future__ import annotations

import unittest

from sol.grouped_sparse_autoencoder import GroupedSparseJewelAutoencoder
from sol.sparse_autoencoder import SparseJewelAutoencoder
from sol.token_grid import GridSpec
from sol.tokenizer_checkpoint import build_tokenizer


class TokenizerCheckpointTests(unittest.TestCase):
    def test_builds_legacy_sparse_tokenizer(self) -> None:
        model = build_tokenizer(
            {
                "architecture": "sparse_variable_count_v1",
                "model_args": {
                    "feature_dim": 22,
                    "model_dim": 16,
                    "latent_dim": 8,
                    "enc_depth": 0,
                    "dec_depth": 0,
                    "heads": 4,
                    "decode_chunk_size": 8,
                    "encoder_mode": "rank",
                    "position_mode": "fourier",
                },
            },
            GridSpec((2, 2, 1), 8),
        )
        self.assertIsInstance(model, SparseJewelAutoencoder)

    def test_builds_grouped_tokenizer(self) -> None:
        model = build_tokenizer(
            {
                "architecture": "grouped_sparse_tokens_v1",
                "model_args": {
                    "feature_dim": 22,
                    "model_dim": 16,
                    "latent_dim": 8,
                    "jewels_per_token": 4,
                    "enc_depth": 0,
                    "dec_depth": 0,
                    "decode_chunk_size": 8,
                },
            },
            GridSpec((2, 2, 1), 8),
        )
        self.assertIsInstance(model, GroupedSparseJewelAutoencoder)

    def test_rejects_unknown_architecture(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported tokenizer architecture"):
            build_tokenizer({"architecture": "future", "model_args": {}}, GridSpec())


if __name__ == "__main__":
    unittest.main()
