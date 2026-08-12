"""Tests for stable-ID oracle-mark topology rollout."""

from __future__ import annotations

import math
import unittest

import torch

from sol.scaffold_topology_data import build_scaffold_topology_views
from sol.scaffold_topology_rollout import (
    oracle_matched_birth_mask,
    rollout_oracle_matched_topology,
)
from sol.token_grid import GridSpec


def _features() -> torch.Tensor:
    features = torch.zeros(48, 22)
    features[:, 0] = torch.linspace(-0.9, 0.9, len(features))
    features[:, 1] = torch.linspace(0.9, -0.9, len(features))
    features[:, 2] = torch.linspace(-0.95, 0.95, len(features))
    features[:, 3] = 2 * math.log(0.08)
    features[:, 6] = 2 * math.log(0.08)
    features[:, 8] = 2 * math.log(0.14)
    features[:, 9:12] = 0.5
    features[:, 21] = 2.0
    return features


class _TargetCountModel:
    def __init__(self, counts: list[torch.Tensor]) -> None:
        self.counts = [value.clone() for value in counts]
        self.index = 0

    def eval(self):
        return self

    def __call__(self, guide, carry):
        return object()

    def decode_counts(self, output, *, occupancy_threshold):
        value = self.counts[self.index].to("cpu")
        self.index += 1
        return value


class ScaffoldTopologyRolloutTests(unittest.TestCase):
    def test_target_counts_recover_ids_density_and_exact_carry(self) -> None:
        features = _features()
        spec = GridSpec((4, 4, 2), 16)
        views = build_scaffold_topology_views(
            features,
            32,
            stride_frames=8,
            support_sigma=2.0,
            grid_spec=spec,
        )
        model = _TargetCountModel([view.births.counts for view in views])
        rollout = rollout_oracle_matched_topology(
            model,
            views,
            [torch.zeros(spec.n_cells, 3) for _ in views],
            features,
            32,
            spec,
            stride_frames=8,
            support_sigma=2.0,
            occupancy_threshold=0.5,
            device="cpu",
        )
        self.assertTrue(rollout.report["stable_ids_unique"])
        self.assertEqual(rollout.report["max_carry_feature_error"], 0.0)
        self.assertAlmostEqual(rollout.report["topology"]["slot_recall"], 1.0)
        self.assertAlmostEqual(
            rollout.report["oracle_retained_density"]["effective_mean_ratio"], 1.0
        )

    def test_match_mask_respects_predicted_rank_prefix(self) -> None:
        spec = GridSpec((4, 4, 2), 16)
        view = build_scaffold_topology_views(
            _features(),
            32,
            stride_frames=8,
            support_sigma=2.0,
            grid_spec=spec,
        )[0]
        predicted = view.births.counts.clone()
        occupied_cell = int(view.births.cell_indices[0])
        predicted[occupied_cell] = 0
        mask = oracle_matched_birth_mask(view, predicted)
        self.assertFalse(mask[view.births.cell_indices == occupied_cell].any())


if __name__ == "__main__":
    unittest.main()
