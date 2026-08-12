"""Tests for sequential scaffold-topology targets and carry rasters."""

from __future__ import annotations

import math
import unittest

import torch

from sol.scaffold_topology_data import (
    build_scaffold_topology_views,
    rasterize_carried_state,
)
from sol.token_grid import GridSpec


def _features() -> torch.Tensor:
    features = torch.zeros(32, 22)
    features[:, 0] = torch.linspace(-0.9, 0.9, len(features))
    features[:, 1] = torch.linspace(0.9, -0.9, len(features))
    features[:, 2] = torch.linspace(-0.95, 0.95, len(features))
    features[:, 3] = 2 * math.log(0.08)
    features[:, 6] = 2 * math.log(0.08)
    features[:, 8] = 2 * math.log(0.14)
    features[:, 9:12] = 0.5
    features[:, 21] = 2.0
    return features


class ScaffoldTopologyDataTests(unittest.TestCase):
    def test_views_include_initial_stride_and_preserve_partition(self) -> None:
        spec = GridSpec((4, 4, 2), 8)
        views = build_scaffold_topology_views(
            _features(),
            32,
            stride_frames=8,
            support_sigma=2.0,
            grid_spec=spec,
        )
        self.assertEqual([view.frontier for view in views], [0, 8, 16, 24])
        self.assertEqual(len(views[0].carried_ids), 0)
        for view in views:
            partition = torch.cat((view.carried_ids, view.births.global_ids)).sort().values
            self.assertTrue(torch.equal(partition, view.active_commit_ids))
            self.assertEqual(int(view.births.counts.sum()), len(view.births.values))
            self.assertTrue(
                torch.equal(view.birth_global_features, _features()[view.births.global_ids])
            )

    def test_carried_raster_is_empty_or_finite_and_bounded(self) -> None:
        spec = GridSpec((4, 4, 2), 8)
        empty = rasterize_carried_state(
            torch.empty(0, 22), 32, 8, 8, spec, support_sigma=2.0
        )
        self.assertEqual(empty.shape, (spec.n_cells, 3))
        self.assertEqual(float(empty.abs().max()), 0.0)

        raster = rasterize_carried_state(
            _features()[:12], 32, 8, 8, spec, support_sigma=2.0
        )
        self.assertTrue(torch.isfinite(raster).all())
        self.assertGreater(float(raster[:, 0].sum()), 0.0)
        self.assertTrue(((raster[:, 1:] >= 0) & (raster[:, 1:] <= 1)).all())


if __name__ == "__main__":
    unittest.main()
