"""Tests for scene-latent prompt training helpers."""

from __future__ import annotations

import unittest

import torch

from sol.scene_latent_prompt_jewel_caster import SceneLatentPromptJewelCaster
from sol.train_scene_latent_prompt_jewel_caster import (
    ScenePromptBatch,
    gaussian_kl,
    scene_control_metrics,
)


class SceneLatentPromptTrainingTests(unittest.TestCase):
    def test_identical_gaussians_have_zero_kl(self) -> None:
        mean = torch.randn(4, 5)
        log_std = torch.randn(4, 5).clamp(-2, 0)
        self.assertAlmostEqual(
            float(gaussian_kl(mean, log_std, mean, log_std)), 0.0, places=6
        )

    def test_controls_cover_density_and_all_active_tokens(self) -> None:
        model = SceneLatentPromptJewelCaster(
            text_dim=8, vocabulary_size=7, scene_dim=5, hidden_dim=16, depth=1
        )
        batch = ScenePromptBatch(
            centers=torch.rand(13, 3) * 2 - 1,
            negative_centers=torch.rand(13, 3) * 2 - 1,
            tokens=torch.randint(0, 7, (13, 3)),
            owners=torch.arange(13) % 3,
        )
        report = scene_control_metrics(
            model, batch, torch.randn(3, 8), torch.randn(3, 8), chunk=5
        )
        self.assertEqual(set(report), {"correct", "shuffled", "null"})
        self.assertEqual(
            set(report["correct"]["token_nll"]),
            {"covariance", "surface", "gradient"},
        )
        self.assertIn("density_nce", report["null"])

    def test_gaussian_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "share a shape"):
            gaussian_kl(
                torch.zeros(2, 3), torch.zeros(2, 3),
                torch.zeros(1, 3), torch.zeros(2, 3),
            )


if __name__ == "__main__":
    unittest.main()
