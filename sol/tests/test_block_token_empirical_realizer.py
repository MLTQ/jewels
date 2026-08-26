"""Tests for empirical macro-Jewel block-token realization."""

from __future__ import annotations

import unittest

import torch

from sol.block_token_empirical_realizer import fit_empirical_block_realizer
from sol.factorized_jewel_casting_language import FactorCodebook, FactorizedCodebook
from sol.jewel_casting_language import CastingNormalizer


def _physical_codebook() -> FactorizedCodebook:
    factors = []
    dimensions = {
        "layout": (0, 1, 2),
        "covariance": (3, 4, 5, 6, 7, 8),
        "surface": (9, 10, 11, 21),
        "gradient": (12, 13, 14, 15, 16, 17, 18, 19, 20),
    }
    for name, dims in dimensions.items():
        factors.append(FactorCodebook(
            name=name, dimensions=dims,
            prototypes=torch.randn(5, 1, len(dims)),
            prototype_count_coordinates=torch.zeros(5),
        ))
    return FactorizedCodebook(
        factors=tuple(factors),
        normalizer=CastingNormalizer(torch.zeros(19), torch.ones(19)),
        grid_shape=(2, 2, 1), bundle_size=1, count_weight=4.0,
    )


class EmpiricalBlockRealizerTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(9)
        self.fields = []
        for _ in range(3):
            field = torch.randn(120, 22, generator=generator)
            field[:, :3] = torch.rand(120, 3, generator=generator) * 2 - 1
            self.fields.append(field)
        self.programs = torch.tensor([
            [0, 1, 2, 3], [1, 1, 2, 3], [0, 2, 2, 3]
        ])
        self.physical = _physical_codebook()

    def test_fit_sample_and_nll_contract(self) -> None:
        realizer, report = fit_empirical_block_realizer(
            self.fields, self.programs,
            physical_codebook=self.physical,
            block_vocabulary_size=4,
        )
        generator = torch.Generator().manual_seed(10)
        centers, tokens = realizer.sample(
            self.programs[0], 97, generator=generator
        )
        self.assertEqual(centers.shape, (97, 3))
        self.assertEqual(tokens.shape, (97, 3))
        self.assertTrue(((centers > -1) & (centers < 1)).all())
        target_tokens = torch.randint(0, 5, (120, 3), generator=generator)
        nll = realizer.token_nll(
            self.programs[0], self.fields[0][:, :3], target_tokens
        )
        self.assertEqual(set(nll["token_nll"]), {"covariance", "surface", "gradient"})
        self.assertEqual(report["training_phrases"], 360)

    def test_prompt_blind_token_must_have_mass(self) -> None:
        realizer, _ = fit_empirical_block_realizer(
            self.fields, self.programs,
            physical_codebook=self.physical,
            block_vocabulary_size=5,
        )
        token = realizer.most_frequent_nonempty_token(self.programs)
        self.assertGreater(float(realizer.mean_jewels_per_occurrence[token]), 0)


if __name__ == "__main__":
    unittest.main()
