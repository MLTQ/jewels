"""Dense tokenizer training-loss tests."""

from __future__ import annotations

import unittest

import torch

from sol.cache_motion_points import _window_motion_points
from sol.corpus import FeatureNormalizer
from sol.synthetic import random_jewels
from sol.token_grid import GridSpec, OccupancyGrid
from sol.train_dense_autoencoder import _sampled_render_loss


class DenseTrainingTests(unittest.TestCase):
    def test_source_motion_pool_balances_frames_and_normalizes_coordinates(self) -> None:
        frames = [torch.zeros(4, 5, 3, dtype=torch.uint8) for _ in range(3)]
        frames[1][2, 3] = 255
        points, scores = _window_motion_points(frames, pool_size=6)
        self.assertEqual(points.shape, (6, 3))
        self.assertEqual(scores.shape, (6,))
        self.assertTrue(((points >= -1) & (points <= 1)).all())
        torch.testing.assert_close(
            points[:, 2], torch.tensor([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0])
        )

    def test_motion_importance_render_loss_is_finite_and_differentiable(self) -> None:
        target_features = random_jewels(12, seed=41)
        normalizer = FeatureNormalizer(
            mean=torch.zeros(22),
            std=torch.ones(22),
        )
        target = OccupancyGrid(GridSpec((2, 2, 2), 12)).pack_compact(
            target_features[None]
        )
        predicted = (target.values + 0.01).detach().requires_grad_()
        loss = _sampled_render_loss(
            predicted,
            target,
            normalizer,
            points_per_example=4,
            motion_fraction=0.5,
            motion_candidate_multiplier=2,
            motion_time_delta=0.1,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(predicted.grad)
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_precomputed_motion_pool_drives_render_loss(self) -> None:
        target_features = random_jewels(12, seed=42)
        normalizer = FeatureNormalizer(torch.zeros(22), torch.ones(22))
        target = OccupancyGrid(GridSpec((2, 2, 2), 12)).pack_compact(
            target_features[None]
        )
        predicted = target.values.detach().clone().requires_grad_()
        pool = torch.tensor(
            [[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]], dtype=torch.float32
        )
        loss = _sampled_render_loss(
            predicted,
            target,
            normalizer,
            points_per_example=2,
            motion_fraction=1.0,
            motion_point_pool=pool,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
