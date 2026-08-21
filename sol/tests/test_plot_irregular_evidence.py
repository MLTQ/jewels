"""Tests for compact irregular-field evidence parsing."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sol.plot_irregular_evidence import evaluation_rows, summary_point


class IrregularEvidencePlotTests(unittest.TestCase):
    def test_evaluation_rows_ignore_training_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.jsonl"
            path.write_text("\n".join([
                json.dumps({"step": 10, "train_psnr": 12.0}),
                json.dumps({
                    "step": 20,
                    "evaluation": {
                        "macro_psnr": 21.0,
                        "structure": {
                            "occupancy_uniformity": 0.98,
                            "active_fraction": 0.6,
                            "mixed_spacetime_tilt_median": 0.4,
                        },
                    },
                }),
            ]))
            self.assertEqual(evaluation_rows(path, step_offset=5), [{
                "step": 25.0,
                "psnr": 21.0,
                "occupancy": 0.98,
                "active": 0.6,
                "mixed_tilt": 0.4,
            }])

    def test_summary_point_reads_held_out_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps({
                "latest_evaluation": {
                    "macro_psnr": 19.5,
                    "structure": {"occupancy_uniformity": 0.97},
                }
            }))
            self.assertEqual(summary_point(path), (19.5, 0.97))


if __name__ == "__main__":
    unittest.main()
