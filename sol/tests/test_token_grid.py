"""Dense-capacity and trainability tests for structured jewel tokens."""

from __future__ import annotations

import unittest

import torch

from sol.autoencoder import StructuredJewelAutoencoder
from sol.synthetic import random_jewels
from sol.token_grid import (
    GridCapacityError,
    GridSpec,
    OccupancyGrid,
)


class TokenGridTests(unittest.TestCase):
    def test_uniform_45k_set_packs_without_loss(self) -> None:
        features = random_jewels(45_000, seed=4)
        grid = OccupancyGrid(GridSpec((8, 8, 4), slots_per_cell=256))
        report = grid.capacity_report(features)
        packed = grid.pack(features)
        self.assertTrue(report.fits)
        self.assertEqual(int(packed.mask.sum()), features.shape[0])

    def test_compact_pack_preserves_every_feature(self) -> None:
        features = random_jewels(200, seed=14)
        grid = OccupancyGrid(GridSpec((4, 4, 2), slots_per_cell=32))
        compact = grid.pack_compact(features)
        self.assertEqual(compact.values.shape, features.shape)
        self.assertEqual(int(compact.counts.sum()), features.shape[0])

    def test_overflow_raises_instead_of_dropping(self) -> None:
        features = random_jewels(9, seed=2)
        features[:, :3] = 0
        grid = OccupancyGrid(GridSpec((1, 1, 1), slots_per_cell=8))
        with self.assertRaises(GridCapacityError):
            grid.pack(features)

    def test_statistics_retain_occupancy(self) -> None:
        one = random_jewels(1, seed=3)
        repeated = one.repeat(7, 1)
        grid = OccupancyGrid(GridSpec((1, 1, 1), slots_per_cell=8))
        stats_one = grid.statistics(one)
        stats_many = grid.statistics(repeated)
        torch.testing.assert_close(stats_one.mean, stats_many.mean)
        self.assertEqual(int(stats_one.count[0]), 1)
        self.assertEqual(int(stats_many.count[0]), 7)

    def test_autoencoder_loss_backpropagates(self) -> None:
        spec = GridSpec((2, 2, 1), slots_per_cell=8)
        features = random_jewels(16, seed=11)[None]
        model = StructuredJewelAutoencoder(
            feature_dim=22,
            model_dim=16,
            latent_dim=8,
            spec=spec,
            enc_depth=1,
            dec_depth=1,
            heads=4,
        )
        loss, terms = model.loss(features)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(terms), {"feature", "existence", "count"})
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
