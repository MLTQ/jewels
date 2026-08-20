"""Tests for the causal temporal-tilt experiment summary."""

from __future__ import annotations

import unittest

import torch

from sol.temporal_tilt_ablation import (
    mean_confidence_interval,
    reconstruction_error,
    summarize,
)


class TemporalTiltAblationTests(unittest.TestCase):
    def test_causal_gate_compares_largest_matched_budget(self) -> None:
        structure_free = {"mixed_spacetime_tilt_median": 0.4}
        structure_axis = {"mixed_spacetime_tilt_median": 0.0}
        records = [
            {
                "geometry_constraint": "free",
                "steps": 100,
                "support_eval_psnr_db": 30.0,
                "n_final": 50,
                "structure": structure_free,
            },
            {
                "geometry_constraint": "axis_aligned",
                "steps": 100,
                "support_eval_psnr_db": 29.0,
                "n_final": 50,
                "structure": structure_axis,
            },
        ]
        report = summarize(records)
        comparison = report["comparisons"][0]
        self.assertEqual(comparison["free_minus_control_psnr_db"], 1.0)
        self.assertTrue(
            all(report["causal_tilt_gate"].values()),
            report["causal_tilt_gate"],
        )

    def test_multiple_seeds_form_distinct_pairs_with_t_interval(self) -> None:
        records = []
        for seed, delta in enumerate((0.6, 0.9, 1.2)):
            for constraint, value, tilt in (
                ("free", 30.0 + delta, 0.4),
                ("axis_aligned", 30.0, 0.0),
            ):
                records.append(
                    {
                        "geometry_constraint": constraint,
                        "steps": 100,
                        "seed": seed,
                        "support_eval_psnr_db": value,
                        "n_final": 50,
                        "structure": {"mixed_spacetime_tilt_median": tilt},
                    }
                )
        report = summarize(records)
        paired = report["largest_budget_axis_aligned"]
        self.assertEqual(paired["pair_count"], 3)
        self.assertAlmostEqual(paired["paired_psnr_delta_db"]["mean"], 0.9)
        self.assertGreater(paired["paired_psnr_delta_db"]["ci95_high"], 0.9)

    def test_confidence_interval_requires_replication(self) -> None:
        single = mean_confidence_interval([1.0])
        self.assertIsNone(single["ci95_low"])
        repeated = mean_confidence_interval([1.0, 1.0, 1.0])
        self.assertEqual(repeated["ci95_low"], 1.0)
        self.assertEqual(repeated["ci95_high"], 1.0)

    def test_error_report_is_zero_for_exact_reconstruction(self) -> None:
        target = torch.rand(3, 4, 5, 3)
        report = reconstruction_error(target, target)
        self.assertTrue(all(value == 0.0 for value in report.values()))


if __name__ == "__main__":
    unittest.main()
