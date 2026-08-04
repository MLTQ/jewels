"""Frozen-tokenizer cache restoration tests."""

from __future__ import annotations

import unittest

from sol.autoencoder import StructuredJewelAutoencoder
from sol.cache_latents import _restore_tokenizer
from sol.sparse_autoencoder import SparseJewelAutoencoder
from sol.token_grid import GridSpec


class CacheLatentsTests(unittest.TestCase):
    def _checkpoint(self, sparse: bool) -> tuple[dict, GridSpec]:
        spec = GridSpec((2, 2, 1), 8)
        model_args = {
            "feature_dim": 22,
            "model_dim": 16,
            "latent_dim": 8,
            "enc_depth": 0,
            "dec_depth": 1,
            "heads": 4,
        }
        model = (
            SparseJewelAutoencoder(
                **model_args, spec=spec, encoder_mode="rank"
            )
            if sparse
            else StructuredJewelAutoencoder(**model_args, spec=spec)
        )
        if sparse:
            model_args["encoder_mode"] = "rank"
        return {
            "model": model.state_dict(),
            "meta": {
                "architecture": (
                    "sparse_variable_count_v1" if sparse else "structured_slots_v1"
                ),
                "model_args": model_args,
            },
        }, spec

    def test_restores_sparse_checkpoint_architecture(self) -> None:
        checkpoint, spec = self._checkpoint(sparse=True)
        self.assertIsInstance(
            _restore_tokenizer(checkpoint, spec), SparseJewelAutoencoder
        )

    def test_restores_structured_checkpoint_architecture(self) -> None:
        checkpoint, spec = self._checkpoint(sparse=False)
        self.assertIsInstance(
            _restore_tokenizer(checkpoint, spec), StructuredJewelAutoencoder
        )


if __name__ == "__main__":
    unittest.main()
