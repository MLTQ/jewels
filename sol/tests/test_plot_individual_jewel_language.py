"""Tests for the individual-Jewel language evidence loader."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sol.plot_individual_jewel_language import load_metrics


class IndividualJewelPlotTests(unittest.TestCase):
    def test_registered_metrics_are_loaded_without_reinterpretation(self) -> None:
        report = {
            "schema": "active-individual-jewel-language-gate-v1",
            "macro": {
                "token_only_voxel_psnr": 22.5,
                "full_residual_voxel_psnr": 120.0,
                "token_only_mixed_tilt_retention": 1.04,
                "token_only_cell_center_lock_fraction": 0.001,
                "eight_frame_decisions": 35000,
            },
            "canonicality": {"summary": {"same_source": 0.45, "different_source": 0.22, "margin": 0.23}},
            "gate": {"passed": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report))
            metrics = load_metrics(path)
        self.assertTrue(metrics["passed"])
        self.assertAlmostEqual(metrics["center_lock_percent"], 0.1)
        self.assertAlmostEqual(metrics["language_margin"], 0.23)

    def test_wrong_report_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({"schema": "other"}))
            with self.assertRaisesRegex(ValueError, "active"):
                load_metrics(path)


if __name__ == "__main__":
    unittest.main()
