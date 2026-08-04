"""Tests for persistent jewel lifecycles and carry/commit rendering."""

from __future__ import annotations

import math
import unittest

import torch

from sol.streaming import build_rolling_windows, measure_jewel_lifecycles
from sol.streaming_metrics import audit_carry_commit_render, measure_streaming_contract


def _features(centers: list[float], sigmas: list[float]) -> torch.Tensor:
    features = torch.zeros(len(centers), 22)
    features[:, 2] = torch.tensor(centers)
    for row, sigma in enumerate(sigmas):
        log_variance = 2 * math.log(sigma)
        features[row, 3] = 2 * math.log(0.15)
        features[row, 6] = 2 * math.log(0.15)
        features[row, 8] = log_variance
        features[row, 9:12] = torch.tensor([0.2 + 0.1 * row, 0.3, 0.4])
        features[row, 21] = 2.0
    return features


class StreamingTests(unittest.TestCase):
    def test_windows_partition_carried_and_new_birth_ids(self) -> None:
        features = _features([-0.75, 0.0, 0.75], [0.5, 0.4, 0.3])
        lifecycles = measure_jewel_lifecycles(features, 9, support_sigma=1.0)
        windows = build_rolling_windows(
            lifecycles, 9, prefix_frames=2, stride_frames=3
        )
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0].birth_ids.tolist(), [0])
        self.assertIn(0, windows[1].carried_ids.tolist())
        self.assertIn(1, windows[1].birth_ids.tolist())
        for window in windows:
            partition = torch.cat((window.carried_ids, window.birth_ids)).sort().values
            self.assertTrue(torch.equal(partition, window.active_commit_ids))

    def test_little_law_identity_uses_observed_lifespans(self) -> None:
        report, _, _ = measure_streaming_contract(
            _features([-0.75, 0.0, 0.75], [0.5, 0.4, 0.3]),
            (9, 10, 20),
            fps=25.0,
            prefix_frames=2,
            stride_frames=3,
            support_sigma=1.0,
        )
        self.assertLess(report["little_law"]["absolute_error"], 1e-6)
        self.assertEqual(report["total_jewels"], 3)
        self.assertEqual(len(report["birth_counts_by_frame"]), 9)

    def test_carry_commit_matches_monolithic_finite_support_render(self) -> None:
        features = _features([-0.75, 0.0, 0.75], [0.5, 0.4, 0.3])
        lifecycles = measure_jewel_lifecycles(features, 9, support_sigma=2.0)
        windows = build_rolling_windows(
            lifecycles, 9, prefix_frames=2, stride_frames=3
        )
        audit = audit_carry_commit_render(
            features,
            9,
            windows,
            support_sigma=2.0,
            points_per_frame=3,
            seed=4,
        )
        self.assertEqual(audit["missing_points"], 0)
        self.assertEqual(audit["duplicate_points"], 0)
        self.assertLess(audit["max_abs_error"], 1e-6)

    def test_rejects_invalid_streaming_arguments(self) -> None:
        features = _features([0.0], [0.2])
        with self.assertRaises(ValueError):
            measure_jewel_lifecycles(features, 1)
        lifecycles = measure_jewel_lifecycles(features, 3)
        with self.assertRaises(ValueError):
            build_rolling_windows(lifecycles, 3, prefix_frames=-1, stride_frames=1)
        with self.assertRaises(ValueError):
            build_rolling_windows(lifecycles, 3, prefix_frames=1, stride_frames=0)


if __name__ == "__main__":
    unittest.main()
