"""Tests for contribution-aware per-frame splat density."""

from __future__ import annotations

import math
import unittest

import torch

from sol.splat_density import measure_frame_splat_density, summarize_counts


def _features(centers: list[float], temporal_sigma: float, weight: float) -> torch.Tensor:
    features = torch.zeros(len(centers), 22)
    features[:, 2] = torch.tensor(centers)
    log_variance = 2 * math.log(temporal_sigma)
    features[:, 3] = 0.0
    features[:, 6] = 0.0
    features[:, 8] = log_variance
    features[:, 21] = math.log(weight / (1 - weight))
    return features


class SplatDensityTests(unittest.TestCase):
    def test_counts_temporal_support_and_peak_alpha(self) -> None:
        features = _features([0.0, 0.9], temporal_sigma=0.1, weight=0.5)
        density = measure_frame_splat_density(
            features,
            3,
            support_sigma=3.0,
            peak_alpha_thresholds=(0.05,),
        )
        self.assertEqual(density.support_counts.tolist(), [0, 1, 1])
        self.assertEqual(density.peak_alpha_counts[0.05].tolist(), [0, 1, 1])
        self.assertAlmostEqual(float(density.effective_peak_alpha_counts[1]), 1.0)

    def test_participation_ratio_counts_equal_contributors(self) -> None:
        density = measure_frame_splat_density(
            _features([0.0, 0.0], temporal_sigma=0.2, weight=0.5),
            1,
        )
        self.assertAlmostEqual(float(density.effective_peak_alpha_counts[0]), 2.0)

    def test_summary_and_invalid_inputs(self) -> None:
        summary = summarize_counts(torch.tensor([1, 2, 7]))
        self.assertAlmostEqual(summary["mean"], 10 / 3, places=6)
        self.assertEqual(
            {key: summary[key] for key in ("median", "min", "max")},
            {"median": 2.0, "min": 1.0, "max": 7.0},
        )
        with self.assertRaises(ValueError):
            measure_frame_splat_density(torch.zeros(1, 22), 0)
        with self.assertRaises(ValueError):
            measure_frame_splat_density(
                torch.zeros(1, 22), 1, peak_alpha_thresholds=(0.0,)
            )


if __name__ == "__main__":
    unittest.main()
