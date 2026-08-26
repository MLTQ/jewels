"""Tests for address-conditioned scene/block constellation realization."""

from __future__ import annotations

import unittest

import torch

from sol.addressed_scene_block_realizer import fit_addressed_scene_block_realizer
from sol.block_token_language import encode_block_tokens, fit_block_token_codebook
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
        normalizer=normalizer, grid_shape=(2, 2, 1), bundle_size=1, count_weight=4.0,
    )


class AddressedSceneBlockRealizerTests(unittest.TestCase):
    def test_nearest_candidates_share_address_and_scene(self) -> None:
        generator = torch.Generator().manual_seed(13)
        fields = []
        for scene in range(2):
            for _ in range(4):
                field = torch.randn(160, 22, generator=generator) + scene
                field[:, :3] = torch.rand(160, 3, generator=generator) * 2 - 1
                fields.append(field)
        normalizer = CastingNormalizer.fit(fields)
        spec = GridSpec((2, 2, 1), 1)
        block_codebook, _ = fit_block_token_codebook(
            fields, normalizer=normalizer, spec=spec,
            vocabulary_size=4, iterations=3,
        )
        programs = torch.stack([
            encode_block_tokens(field, block_codebook)[0] for field in fields
        ])
        realizer, report = fit_addressed_scene_block_realizer(
            fields, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            block_codebook=block_codebook,
            physical_codebook=_physical(normalizer),
        )
        nearest, _ = realizer.nearest_fields(1, programs[4], 4)
        self.assertTrue((nearest >= 4).all())
        centers, tokens, stats = realizer.sample(
            1, programs[4], 137, generator=generator
        )
        self.assertEqual(centers.shape, (137, 3))
        self.assertEqual(tokens.shape, (137, 3))
        self.assertEqual(stats["scene_token"], 1)
        self.assertEqual(report["same_address_candidates_per_semantic_scene"], {"0": 4, "1": 4})


if __name__ == "__main__":
    unittest.main()
