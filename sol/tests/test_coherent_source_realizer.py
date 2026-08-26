"""Tests for coherent source-level Jewel realization."""

from __future__ import annotations

import unittest

import torch

from sol.block_token_language import encode_block_tokens, fit_block_token_codebook
from sol.coherent_source_realizer import fit_coherent_source_realizer
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


class CoherentSourceRealizerTests(unittest.TestCase):
    def test_selection_is_one_complete_scene_eligible_field(self) -> None:
        generator = torch.Generator().manual_seed(17)
        fields = []
        for scene in range(2):
            for source in range(3):
                field = torch.randn(96, 22, generator=generator) + scene * 0.2
                field[:, :3] = torch.rand(96, 3, generator=generator) * 2 - 1
                field[:, 9:12] += source * 0.1
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
        realizer, report = fit_coherent_source_realizer(
            fields, torch.tensor([0, 0, 0, 1, 1, 1]),
            block_codebook=block_codebook,
            physical_codebook=_physical(normalizer),
            jitter_std=0.0,
        )
        selected, distance = realizer.select_source(1, programs[4])
        self.assertIn(selected, (3, 4, 5))
        self.assertGreaterEqual(distance, 0.0)
        centers, tokens, stats = realizer.sample(
            1, programs[4], 96, generator=generator
        )
        self.assertTrue(torch.equal(centers, fields[selected][:, :3]))
        self.assertEqual(tokens.shape, (96, 3))
        self.assertEqual(stats["selected_training_field"], selected)
        self.assertEqual(stats["emitted_jewels"], 96)
        self.assertTrue(report["one_source_choice_per_window"])

    def test_rejects_wrong_program_shape(self) -> None:
        generator = torch.Generator().manual_seed(23)
        fields = [torch.randn(32, 22, generator=generator) for _ in range(2)]
        for field in fields:
            field[:, :3] = torch.rand(32, 3, generator=generator) * 2 - 1
        normalizer = CastingNormalizer.fit(fields)
        spec = GridSpec((2, 1, 1), 1)
        block_codebook, _ = fit_block_token_codebook(
            fields, normalizer=normalizer, spec=spec,
            vocabulary_size=2, iterations=2,
        )
        realizer, _ = fit_coherent_source_realizer(
            fields, torch.tensor([0, 0]), block_codebook=block_codebook,
            physical_codebook=_physical(normalizer),
        )
        with self.assertRaisesRegex(ValueError, "one token"):
            realizer.select_source(0, torch.zeros(3, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
