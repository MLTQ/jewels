"""Tests for the discrete local spacetime block language."""

from __future__ import annotations

import unittest

import torch

from sol.block_token_language import (
    block_descriptors,
    block_local_coordinates,
    block_serialization_order,
    encode_block_tokens,
    fit_block_token_codebook,
    most_frequent_block_token,
)
from sol.jewel_casting_language import CastingNormalizer
from sol.token_grid import GridSpec


class BlockTokenLanguageTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(7)
        self.fields = []
        for _ in range(3):
            field = torch.randn(400, 22, generator=generator)
            field[:, :3] = torch.rand(400, 3, generator=generator) * 2 - 1
            self.fields.append(field)
        self.normalizer = CastingNormalizer.fit(self.fields)
        self.spec = GridSpec((4, 4, 2), slots_per_cell=1)

    def test_local_coordinates_preserve_continuous_offsets(self) -> None:
        centers = torch.tensor([[-0.91, -0.37, 0.42], [0.13, 0.77, -0.24]])
        cells, local = block_local_coordinates(centers, self.spec)
        self.assertEqual(cells.shape, (2,))
        self.assertTrue(((local >= -1) & (local <= 1)).all())
        self.assertGreater(float(local.std()), 0.05)

    def test_descriptor_has_frozen_dimension_and_keeps_all_blocks(self) -> None:
        descriptors = block_descriptors(
            self.fields[0], spec=self.spec,
            intrinsic_mean=self.normalizer.intrinsic_mean,
            intrinsic_std=self.normalizer.intrinsic_std,
        )
        self.assertEqual(descriptors.shape, (self.spec.n_cells, 77))
        self.assertTrue(torch.isfinite(descriptors).all())

    def test_fit_and_encode_emit_one_token_per_block(self) -> None:
        codebook, report = fit_block_token_codebook(
            self.fields, normalizer=self.normalizer, spec=self.spec,
            vocabulary_size=8, iterations=3,
        )
        tokens, distances = encode_block_tokens(self.fields[0], codebook)
        self.assertEqual(tokens.shape, (self.spec.n_cells,))
        self.assertEqual(distances.shape, (self.spec.n_cells,))
        self.assertEqual(report["descriptor_dim"], 77)
        self.assertLess(int(tokens.max()), 8)

    def test_time_major_morton_order_is_a_permutation(self) -> None:
        order = block_serialization_order(GridSpec((8, 8, 4), 1))
        self.assertEqual(len(order), 256)
        self.assertEqual(sorted(order.tolist()), list(range(256)))
        self.assertTrue((order[:64] % 4 == 0).all())

    def test_most_frequent_token_is_deterministic(self) -> None:
        programs = torch.tensor([[1, 1, 2], [2, 1, 3]])
        self.assertEqual(most_frequent_block_token(programs, 5), 1)


if __name__ == "__main__":
    unittest.main()
