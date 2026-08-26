"""Tests for semantic density-balanced trajectory realization."""

from __future__ import annotations

import unittest

import torch

from sol.block_token_language import encode_block_tokens, fit_block_token_codebook
from sol.factorized_jewel_casting_language import FactorCodebook, FactorizedCodebook
from sol.jewel_casting_language import CastingNormalizer
from sol.semantic_trajectory_realizer import fit_semantic_trajectory_realizer
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
            prototypes=torch.randn(7, 1, len(dims)),
            prototype_count_coordinates=torch.zeros(7),
        ) for name, dims in dimensions.items()),
        normalizer=normalizer, grid_shape=(2, 2, 2), bundle_size=1, count_weight=4.0,
    )


class SemanticTrajectoryRealizerTests(unittest.TestCase):
    def test_semantic_path_and_density_balance(self) -> None:
        generator = torch.Generator().manual_seed(37)
        fields = []
        for scene in range(2):
            for source in range(3):
                field = torch.randn(512, 22, generator=generator) + scene * 0.1
                field[:, :3] = torch.rand(512, 3, generator=generator) * 2 - 1
                field[:, 9:12] += scene * 0.5 + source * 0.05
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
        realizer, report = fit_semantic_trajectory_realizer(
            fields, torch.tensor([0, 0, 0, 1, 1, 1]),
            block_codebook=block_codebook,
            physical_codebook=_physical(normalizer),
            jitter_std=0.0,
        )
        centers, tokens, stats = realizer.sample(
            0, programs[0], 512, generator=generator
        )
        self.assertEqual(realizer.scene_paths.shape, (2, 2, 2))
        self.assertNotEqual(
            stats["foreground_training_field"], stats["background_training_field"]
        )
        self.assertGreaterEqual(stats["foreground_fraction"], 0.20)
        self.assertGreaterEqual(stats["background_fraction"], 0.20)
        self.assertEqual(centers.shape, (512, 3))
        self.assertEqual(tokens.shape, (512, 3))
        self.assertTrue(report["density_balanced_boundary"])
        _, _, wrong = realizer.sample(
            0, programs[0], 512, generator=generator, foreground_scene_token=1
        )
        self.assertGreaterEqual(wrong["foreground_training_field"], 3)
        self.assertLess(wrong["background_training_field"], 3)
        self.assertEqual(wrong["tube_centers"], stats["tube_centers"])
        balanced_centers, balanced_tokens, balanced = (
            realizer.sample_rank_balanced_from_donors(
                0, 0, 1, 512, generator=generator
            )
        )
        self.assertEqual(balanced_centers.shape, (512, 3))
        self.assertEqual(balanced_tokens.shape, (512, 3))
        self.assertEqual(balanced["foreground_fraction"], 0.5)
        self.assertEqual(balanced["background_fraction"], 0.5)
        self.assertEqual(balanced["adjustment_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
