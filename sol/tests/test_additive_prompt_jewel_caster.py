"""Tests for additive style/action Jewel language modeling."""

from __future__ import annotations

import unittest

import torch

from sol.additive_prompt_jewel_caster import (
    AdditivePromptJewelCaster,
    accumulate_language_counts,
)
from sol.token_grid import GridSpec


class AdditivePromptJewelCasterTests(unittest.TestCase):
    def test_probabilities_normalize_and_sampling_is_irregular(self) -> None:
        spec = GridSpec((2, 2, 2), slots_per_cell=1)
        samples = []
        for style in range(2):
            for action in range(2):
                centers = torch.rand(128, 3) * 2 - 1
                tokens = torch.randint(0, 8, (128, 3))
                samples.append((centers, tokens, style, action))
        counts = accumulate_language_counts(
            samples, spec=spec, vocabulary_size=8,
            style_count=2, action_count=2,
        )
        model = AdditivePromptJewelCaster(counts, spec=spec)
        cells, tokens = model.probabilities(0, 1)
        torch.testing.assert_close(cells.sum(), torch.tensor(1.0))
        torch.testing.assert_close(
            tokens.sum(dim=2), torch.ones(spec.n_cells, 3)
        )
        generator = torch.Generator().manual_seed(8)
        centers, marks = model.sample(
            97, style_index=0, action_index=1, generator=generator, chunk=17
        )
        self.assertEqual(centers.shape, (97, 3))
        self.assertEqual(marks.shape, (97, 3))
        self.assertGreater(float(centers.std()), 0.1)


if __name__ == "__main__":
    unittest.main()
