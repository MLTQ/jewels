"""Tests for multi-source temporal-tilt replication aggregation."""

from __future__ import annotations

import unittest

from sol.aggregate_temporal_tilt import aggregate_reports


def source_report(source: str) -> dict:
    """Build three passing paired comparisons for one source."""
    comparisons = []
    for seed, delta in enumerate((0.7, 0.8, 0.9)):
        comparisons.append(
            {
                "steps": 300,
                "seed": seed,
                "control": "axis_aligned",
                "free_minus_control_psnr_db": delta,
                "free_minus_control_primitives": 0,
                "free_parameter_bytes": 4000,
                "control_parameter_bytes": 4000,
                "free_mixed_tilt_median": 0.4,
                "control_mixed_tilt_median": 0.0,
                "control_minus_free_rgb_mae": 0.01,
                "control_minus_free_motion_top20_rgb_mae": 0.02,
                "free_psnr_db_per_1000_primitives": 60.0,
                "control_psnr_db_per_1000_primitives": 58.0,
                "free_psnr_db_per_parameter_megabyte": 6000.0,
                "control_psnr_db_per_parameter_megabyte": 5800.0,
            }
        )
    return {
        "schema": "temporal-tilt-ablation-v2",
        "source": source,
        "protocol": {"steps": [300]},
        "summary": {
            "comparisons": comparisons,
            "largest_budget_axis_aligned": {"steps": 300},
        },
    }


class AggregateTemporalTiltTests(unittest.TestCase):
    def test_three_sources_and_nine_pairs_pass_decision_gate(self) -> None:
        reports = [source_report(f"/clips/source_{index}.avi") for index in range(3)]
        aggregate = aggregate_reports(reports)
        self.assertEqual(aggregate["source_count"], 3)
        self.assertEqual(aggregate["pair_count"], 9)
        self.assertTrue(all(aggregate["decision_gate"].values()))
        self.assertGreater(
            aggregate["aggregate"]["paired_psnr_delta_db"]["ci95_low"], 0.0
        )

    def test_protocol_mismatch_fails_loudly(self) -> None:
        reports = [source_report("a.avi"), source_report("b.avi")]
        reports[1]["protocol"] = {"steps": [900]}
        with self.assertRaisesRegex(ValueError, "same protocol"):
            aggregate_reports(reports)


if __name__ == "__main__":
    unittest.main()
