"""Frozen latent-cache validation and normalization tests."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from sol.latent_data import LatentCache, load_latent_cache, save_latent_cache


class LatentDataTests(unittest.TestCase):
    def _cache(self) -> LatentCache:
        latents = torch.randn(4, 3, 2)
        train_mask = torch.tensor([True, True, False, False])
        training = latents[train_mask]
        conditions = torch.randn(4, 5)
        training_conditions = conditions[train_mask]
        return LatentCache(
            latents=latents,
            conditions=conditions,
            names=("a0", "a1", "b0", "b1"),
            source_ids=("a", "a", "b", "b"),
            train_mask=train_mask,
            latent_mean=training.mean(0),
            latent_std=training.std(0).clamp_min(1e-3),
            condition_mean=training_conditions.mean(0),
            condition_std=training_conditions.std(0).clamp_min(1e-3),
            metadata={"tokenizer_step": 1},
        )

    def test_normalization_uses_training_samples_and_roundtrips(self) -> None:
        cache = self._cache()
        normalized, _, _ = cache.split(train=True)
        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=5)
        _, normalized_conditions, _ = cache.split(train=True)
        self.assertAlmostEqual(float(normalized_conditions.mean()), 0.0, places=5)
        restored = cache.denormalize(cache.normalized_latents)
        torch.testing.assert_close(restored, cache.latents)

    def test_atomic_save_load_preserves_contract(self) -> None:
        cache = self._cache()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            save_latent_cache(cache, path)
            restored = load_latent_cache(path)
        torch.testing.assert_close(restored.latents, cache.latents)
        self.assertEqual(restored.source_ids, cache.source_ids)

    def test_legacy_cache_uses_identity_condition_transform(self) -> None:
        state = self._cache().state_dict()
        del state["condition_mean"]
        del state["condition_std"]
        restored = LatentCache.from_state_dict(state)
        torch.testing.assert_close(restored.normalized_conditions, restored.conditions)

    def test_rejects_source_leakage(self) -> None:
        cache = self._cache()
        with self.assertRaises(ValueError):
            LatentCache(
                latents=cache.latents,
                conditions=cache.conditions,
                names=cache.names,
                source_ids=("a", "b", "a", "b"),
                train_mask=cache.train_mask,
                latent_mean=cache.latent_mean,
                latent_std=cache.latent_std,
                condition_mean=cache.condition_mean,
                condition_std=cache.condition_std,
                metadata={},
            )


if __name__ == "__main__":
    unittest.main()
