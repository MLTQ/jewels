"""Tests for native prompt-conditioned Jewel casting contracts."""

from __future__ import annotations

import unittest

import torch

from sol.factorized_jewel_casting_language import fit_factorized_codebook
from sol.prompt_jewel_caster import (
    FactorizedPromptJewelCaster,
    PromptJewelCaster,
    active_cell_histogram,
    active_tokens_to_features,
    encode_active_jewel_tokens,
)
from sol.synthetic import random_jewels
from sol.token_grid import GridSpec


class PromptJewelCasterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = GridSpec((2, 2, 2), slots_per_cell=1)
        fields = [random_jewels(96, seed=seed) for seed in range(3)]
        self.field = fields[2]
        self.codebook, _ = fit_factorized_codebook(
            fields[:2], spec=self.spec, bundle_size=1,
            vocabulary_size=8, iterations=2, assignment_chunk=16,
        )

    def test_active_token_decode_preserves_continuous_centers(self) -> None:
        tokens = encode_active_jewel_tokens(self.field, self.codebook)
        decoded = active_tokens_to_features(
            self.field[:, :3], tokens, self.codebook
        )
        torch.testing.assert_close(decoded[:, :3], self.field[:, :3])
        histogram = active_cell_histogram(
            decoded[:, :3], tokens, spec=self.spec,
            vocabulary_size=self.codebook.vocabulary_size,
        )
        self.assertEqual(histogram.numel(), self.spec.n_cells * 8 * 3)

    def test_prompt_caster_loss_and_free_sampling_shapes(self) -> None:
        model = PromptJewelCaster(
            text_dim=16, vocabulary_size=8, hidden_dim=32,
            depth=2, mixture_components=4,
        )
        text = torch.randn(12, 16)
        centers = torch.rand(12, 3) * 2 - 1
        tokens = torch.randint(0, 8, (12, 3))
        loss, metrics = model.loss(text, centers, tokens)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(metrics), {"token_nll", "centroid_nll"})
        prompt = text[:1]
        generator = torch.Generator().manual_seed(4)
        sampled_centers = model.centroid_density.sample(
            prompt, 17, generator=generator
        )
        sampled_tokens = model.sample_tokens(
            prompt, sampled_centers, generator=generator, top_k=4, chunk=5
        )
        self.assertEqual(sampled_centers.shape, (17, 3))
        self.assertEqual(sampled_tokens.shape, (17, 3))

    def test_factorized_caster_samples_continuous_centers_and_tokens(self) -> None:
        model = FactorizedPromptJewelCaster(
            text_dim=16, vocabulary_size=8, hidden_dim=32, depth=2
        )
        style, action = torch.randn(1, 16), torch.randn(1, 16)
        generator = torch.Generator().manual_seed(9)
        centers = model.sample_centers(
            style, action, 19, generator=generator, proposal_multiplier=2, chunk=7
        )
        tokens = model.sample_tokens(
            style, action, centers, generator=generator, top_k=4, chunk=6
        )
        negative = torch.rand_like(centers) * 2 - 1
        loss, metrics = model.loss(
            style.expand(19, -1), action.expand(19, -1),
            centers, tokens, negative,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(metrics), {"token_nll", "density_nce"})
        self.assertEqual(tokens.shape, (19, 3))


if __name__ == "__main__":
    unittest.main()
