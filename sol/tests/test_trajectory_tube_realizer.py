"""Tests for compositional trajectory-tube Jewel realization."""

from __future__ import annotations

import unittest

import torch

from sol.block_token_language import encode_block_tokens, fit_block_token_codebook
from sol.factorized_jewel_casting_language import FactorCodebook, FactorizedCodebook
from sol.jewel_casting_language import CastingNormalizer
from sol.token_grid import GridSpec
from sol.trajectory_tube_realizer import fit_trajectory_tube_realizer


def _physical(normalizer: CastingNormalizer) -> FactorizedCodebook:
    dimensions = {
        "layout": (0, 1, 2), "covariance": (3, 4, 5, 6, 7, 8),
        "surface": (9, 10, 11, 21),
        "gradient": (12, 13, 14, 15, 16, 17, 18, 19, 20),
    }
    return FactorizedCodebook(
        factors=tuple(FactorCodebook(
            name=name, dimensions=dims,
            prototypes=torch.randn(7, 1, len(dims)),
            prototype_count_coordinates=torch.zeros(7),
        ) for name, dims in dimensions.items()),
        normalizer=normalizer, grid_shape=(2, 2, 2), bundle_size=1, count_weight=4.0,
    )


class TrajectoryTubeRealizerTests(unittest.TestCase):
    def test_composite_uses_two_sources_and_exact_count(self) -> None:
        generator = torch.Generator().manual_seed(31)
        fields = []
        for scene in range(2):
            for source in range(3):
                field = torch.randn(256, 22, generator=generator) + scene * 0.1
                field[:, :3] = torch.rand(256, 3, generator=generator) * 2 - 1
                field[:, 9:12] += source * 0.2
                fields.append(field)
        normalizer = CastingNormalizer.fit(fields)
        spec = GridSpec((2, 2, 2), 1)
        block_codebook, _ = fit_block_token_codebook(
            fields, normalizer=normalizer, spec=spec,
            vocabulary_size=6, iterations=3,
        )
        programs = torch.stack([
            encode_block_tokens(field, block_codebook)[0] for field in fields
        ])
        realizer, report = fit_trajectory_tube_realizer(
            fields, torch.tensor([0, 0, 0, 1, 1, 1]),
            block_codebook=block_codebook,
            physical_codebook=_physical(normalizer),
            jitter_std=0.0,
        )
        centers, tokens, stats = realizer.sample(
            0, programs[0], 256, generator=generator
        )
        self.assertNotEqual(
            stats["foreground_training_field"], stats["background_training_field"]
        )
        self.assertEqual(centers.shape, (256, 3))
        self.assertEqual(tokens.shape, (256, 3))
        self.assertEqual(stats["emitted_jewels"], 256)
        self.assertGreater(stats["foreground_fraction"], 0.1)
        self.assertGreater(stats["background_fraction"], 0.1)
        self.assertEqual(len(stats["tube_centers"]), 2)
        self.assertTrue(report["foreground_background_are_distinct"])
        _, _, wrong = realizer.sample(
            0, programs[0], 256, generator=generator, foreground_scene_token=1
        )
        self.assertGreaterEqual(wrong["foreground_training_field"], 3)
        self.assertLess(wrong["background_training_field"], 3)


if __name__ == "__main__":
    unittest.main()
