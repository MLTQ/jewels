"""Tests for support renderer benchmark summary logic."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmark_support_renderer.py"
SPEC = importlib.util.spec_from_file_location("benchmark_support_renderer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SupportRendererBenchmarkTests(unittest.TestCase):
    def test_summary_requires_correctness_and_ratio_gate(self) -> None:
        records = [
            {"primitive_count": 10, "mode": "knn", "seconds_median": 1.0},
            {
                "primitive_count": 10,
                "mode": "support_tiled",
                "seconds_median": 1.5,
                "correctness": {"max_abs": 1e-6},
            },
        ]
        summary = MODULE.summarize(records, ratio_gate=2.0)
        self.assertTrue(summary["all_requested_modes_completed"])
        self.assertTrue(summary["correctness_max_abs_below_2e_5"])
        self.assertTrue(summary["within_ratio_gate_at_all_shared_scales"])

        records[1]["seconds_median"] = 2.1
        records[1]["correctness"]["max_abs"] = 3e-5
        summary = MODULE.summarize(records, ratio_gate=2.0)
        self.assertFalse(summary["correctness_max_abs_below_2e_5"])
        self.assertFalse(summary["within_ratio_gate_at_all_shared_scales"])

    def test_summary_treats_recorded_error_as_incomplete(self) -> None:
        records = [
            {"primitive_count": 10, "mode": "knn", "seconds_median": 1.0},
            {"primitive_count": 10, "mode": "support_tiled", "error": "OOM"},
        ]
        summary = MODULE.summarize(records, ratio_gate=2.0)
        self.assertFalse(summary["all_requested_modes_completed"])
        self.assertFalse(summary["correctness_max_abs_below_2e_5"])
        self.assertFalse(summary["within_ratio_gate_at_all_shared_scales"])


if __name__ == "__main__":
    unittest.main()
