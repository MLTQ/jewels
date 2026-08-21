"""Tests for factorized-v3 evidence parsing."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sol.plot_factorized_evidence import progression_rows, size_weight, summary_metrics


class FactorizedEvidencePlotTests(unittest.TestCase):
    def test_summary_metrics_read_held_out_structure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps({
                "jewels_per_window": 40960,
                "latest_evaluation": {
                    "macro_psnr": 18.2,
                    "structure": {
                        "occupancy_uniformity": 0.97,
                        "active_fraction": 0.63,
                        "mixed_spacetime_tilt_median": 0.51,
                        "extent_median": 0.031,
                    },
                },
            }))
            self.assertEqual(summary_metrics(path), {
                "proposals": 40960.0,
                "psnr": 18.2,
                "occupancy": 0.97,
                "active": 0.63,
                "tilt": 0.51,
                "extent": 0.031,
            })

    def test_progression_ignores_training_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_log.jsonl"
            path.write_text("\n".join([
                json.dumps({"step": 5, "train_psnr": 10}),
                json.dumps({"step": 20, "evaluation": {"macro_psnr": 17.5}}),
            ]))
            self.assertEqual(progression_rows(path, 600), [{"step": 620.0, "psnr": 17.5}])

    def test_size_weight_decodes_registered_names(self) -> None:
        self.assertEqual(size_weight(Path("control_seed0_600/summary.json")), 0.0)
        self.assertEqual(size_weight(Path("size003_offset035_seed0_600/summary.json")), 0.03)
        self.assertEqual(size_weight(Path("size0023_offset035_seed0_600/summary.json")), 0.023)
        with self.assertRaises(ValueError):
            size_weight(Path("unregistered/summary.json"))


if __name__ == "__main__":
    unittest.main()
