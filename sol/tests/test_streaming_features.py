"""Tests for physical-time continuation feature transforms."""

from __future__ import annotations

import math
import unittest

import torch

from sol.render import render_exact
from sol.streaming_features import to_frontier_time, to_global_time


def _features() -> torch.Tensor:
    features = torch.zeros(3, 22)
    features[:, :3] = torch.tensor(
        [[-0.4, 0.1, -0.5], [0.2, -0.3, 0.0], [0.5, 0.4, 0.6]]
    )
    features[:, 3] = math.log(0.08)
    features[:, 6] = math.log(0.05)
    features[:, 8] = math.log(0.12)
    features[:, 4] = torch.tensor([0.01, -0.015, 0.005])
    features[:, 5] = torch.tensor([0.02, 0.01, -0.01])
    features[:, 7] = torch.tensor([-0.01, 0.015, 0.02])
    features[:, 9:12] = torch.tensor([0.3, 0.5, 0.7])
    features[:, 12:21] = torch.arange(27).reshape(3, 9) * 0.002
    features[:, 21] = 1.5
    return features


class StreamingFeatureTests(unittest.TestCase):
    def test_round_trip_restores_canonical_features(self) -> None:
        features = _features()
        local = to_frontier_time(features, 96, 48, 16)
        restored = to_global_time(local, 96, 48, 16)
        self.assertLess(float((restored - features).abs().max()), 2e-5)

    def test_render_is_invariant_under_time_reframing(self) -> None:
        features = _features()
        local = to_frontier_time(features, 96, 48, 16)
        points = torch.tensor(
            [[-0.4, 0.1, -0.5], [0.2, -0.3, 0.0], [0.5, 0.4, 0.6]]
        )
        local_points = points.clone()
        scale = 95 / 32
        offset = (47.5 - 48) / 16
        local_points[:, 2] = points[:, 2] * scale + offset
        reference = render_exact(features, points)
        candidate = render_exact(local, local_points)
        self.assertLess(float((candidate - reference).abs().max()), 2e-5)


if __name__ == "__main__":
    unittest.main()
