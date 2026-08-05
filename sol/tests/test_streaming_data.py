"""Tests for stable-ID prefix/future continuation targets."""

from __future__ import annotations

import math
import unittest

import torch

from sol.streaming_data import build_continuation_dataset, rasterize_context
from sol.token_grid import GridSpec


def _features() -> torch.Tensor:
    centers = torch.linspace(-0.9, 0.9, 24)
    features = torch.zeros(len(centers), 22)
    features[:, 0] = torch.linspace(-0.8, 0.8, len(centers))
    features[:, 1] = torch.linspace(0.8, -0.8, len(centers))
    features[:, 2] = centers
    features[:, 3] = 2 * math.log(0.08)
    features[:, 6] = 2 * math.log(0.08)
    features[:, 8] = 2 * math.log(0.16)
    features[:, 9:12] = 0.4
    features[:, 21] = 2.0
    return features


class StreamingDataTests(unittest.TestCase):
    def test_views_preserve_ids_and_partition_future_state(self) -> None:
        data = build_continuation_dataset(
            _features(),
            32,
            prefix_frames=8,
            stride_frames=4,
            support_sigma=2.0,
            grid_spec=GridSpec((4, 4, 2), 8),
        )
        self.assertEqual(len(data.views), 6)
        for view in data.views:
            ids = torch.cat((view.carried_ids, view.births.global_ids)).sort().values
            self.assertTrue(torch.equal(ids, view.active_commit_ids))
            self.assertEqual(int(view.births.counts.sum()), len(view.births.values))
            self.assertTrue((view.births.slot_indices < 8).all())

    def test_separate_standardizers_round_trip(self) -> None:
        data = build_continuation_dataset(
            _features(),
            32,
            prefix_frames=8,
            stride_frames=4,
            support_sigma=2.0,
            grid_spec=GridSpec((4, 4, 2), 8),
        )
        values = data.views[0].births.values
        restored = data.birth_standardizer.denormalize(
            data.birth_standardizer.normalize(values)
        )
        self.assertLess(float((restored - values).abs().max()), 1e-6)

    def test_context_raster_preserves_total_occupancy(self) -> None:
        data = build_continuation_dataset(
            _features(),
            32,
            prefix_frames=8,
            stride_frames=4,
            support_sigma=2.0,
            grid_spec=GridSpec((4, 4, 2), 8),
        )
        view = data.views[0]
        raster = rasterize_context(
            view.context_features,
            data.context_standardizer,
            prefix_frames=data.prefix_frames,
            stride_frames=data.stride_frames,
            grid_shape=data.grid_spec.shape,
        )
        self.assertEqual(raster.shape, (32, 46))
        occupied = raster[:, -1]
        self.assertGreater(int(occupied.sum()), 0)
        self.assertLessEqual(int(occupied.sum()), len(view.context_features))


if __name__ == "__main__":
    unittest.main()
