"""Tests for the shared-scene native Jewel caster."""

from __future__ import annotations

import unittest

import torch

from sol.scene_latent_prompt_jewel_caster import SceneLatentPromptJewelCaster


class SceneLatentPromptJewelCasterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SceneLatentPromptJewelCaster(
            text_dim=8, vocabulary_size=7, scene_dim=5, hidden_dim=16, depth=1
        )

    def test_prior_and_heads_have_expected_shapes(self) -> None:
        style, action = torch.randn(3, 8), torch.randn(3, 8)
        mean, log_std = self.model.prior_parameters(style, action)
        self.assertEqual(mean.shape, (3, 5))
        self.assertEqual(log_std.shape, (3, 5))
        centers = torch.rand(3, 3) * 2 - 1
        self.assertEqual(
            self.model.token_logits(style, action, mean, centers).shape,
            (3, 3, 7),
        )
        self.assertEqual(
            self.model.intensity_logits(style, action, mean, centers).shape,
            (3,),
        )

    def test_one_scene_is_shared_across_all_sampled_jewels(self) -> None:
        style, action = torch.randn(1, 8), torch.randn(1, 8)
        generator = torch.Generator().manual_seed(4)
        scene = self.model.sample_scene(style, action, generator=generator)
        centers = self.model.sample_centers(
            style, action, scene, 11, generator=generator, proposal_multiplier=2
        )
        tokens = self.model.sample_tokens(
            style, action, scene, centers, generator=generator, top_k=3
        )
        self.assertEqual(centers.shape, (11, 3))
        self.assertEqual(tokens.shape, (11, 3))

    def test_rejects_row_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "align"):
            self.model.token_logits(
                torch.randn(2, 8), torch.randn(2, 8),
                torch.randn(1, 5), torch.randn(2, 3),
            )


if __name__ == "__main__":
    unittest.main()
