"""Tests for the causal temporal-tilt experiment summary."""

from __future__ import annotations

import unittest

from sol.temporal_tilt_ablation import summarize


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


if __name__ == "__main__":
    unittest.main()
