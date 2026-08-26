"""Tests for complete medoid block-constellation realization."""

from __future__ import annotations

import unittest

import torch

from sol.block_token_constellation_realizer import fit_constellation_block_realizer
from sol.block_token_language import fit_block_token_codebook
from sol.factorized_jewel_casting_language import FactorCodebook, FactorizedCodebook
from sol.jewel_casting_language import CastingNormalizer
from sol.token_grid import GridSpec


def _physical(normalizer: CastingNormalizer) -> FactorizedCodebook:
    dimensions = {
        "layout": (0, 1, 2), "covariance": (3, 4, 5, 6, 7, 8),
        "surface": (9, 10, 11, 21),
        "gradient": (12, 13, 14, 15, 16, 17, 18, 19, 20),
    }
    return FactorizedCodebook(
        factors=tuple(FactorCodebook(
            name=name, dimensions=dims,
            prototypes=torch.randn(5, 1, len(dims)),
            prototype_count_coordinates=torch.zeros(5),
        ) for name, dims in dimensions.items()),
        normalizer=normalizer, grid_shape=(2, 2, 1),
        bundle_size=1, count_weight=4.0,
    )


class ConstellationBlockRealizerTests(unittest.TestCase):
    def test_medoid_templates_cast_exact_requested_count(self) -> None:
        generator = torch.Generator().manual_seed(11)
        fields = []
        for _ in range(4):
            field = torch.randn(160, 22, generator=generator)
            field[:, :3] = torch.rand(160, 3, generator=generator) * 2 - 1
            fields.append(field)
        normalizer = CastingNormalizer.fit(fields)
        spec = GridSpec((2, 2, 1), 1)
        block_codebook, _ = fit_block_token_codebook(
            fields, normalizer=normalizer, spec=spec,
            vocabulary_size=4, iterations=3,
        )
        from sol.block_token_language import encode_block_tokens
        programs = torch.stack([
            encode_block_tokens(field, block_codebook)[0] for field in fields
        ])
        realizer, report = fit_constellation_block_realizer(
            fields, programs, block_codebook=block_codebook,
            physical_codebook=_physical(normalizer),
        )
        centers, tokens, stats = realizer.sample(
            programs[0], 137, generator=generator
        )
        self.assertEqual(centers.shape, (137, 3))
        self.assertEqual(tokens.shape, (137, 3))
        self.assertEqual(stats["requested_jewels"], 137)
        self.assertEqual(report["vocabulary_size"], 4)
        self.assertTrue(((centers > -1) & (centers < 1)).all())


if __name__ == "__main__":
    unittest.main()
