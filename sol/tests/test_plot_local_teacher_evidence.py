"""Tests for local-teacher evidence parsing."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sol.plot_local_teacher_evidence import load_exact_metrics, load_screen_metrics


class LocalTeacherEvidencePlotTests(unittest.TestCase):
    def test_load_screen_metrics_reads_final_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, arm in enumerate(("control", "appearance", "full")):
                run = root / f"{arm}_seed0_600"
                run.mkdir()
                (run / "summary.json").write_text(json.dumps({
                    "latest_evaluation": {
                        "macro_psnr": 18.0 + index,
                        "structure": {
                            "occupancy_uniformity": 0.98 + index / 1000,
                            "active_fraction": 0.60 + index / 100,
                            "mixed_spacetime_tilt_median": 0.50 + index / 100,
                            "extent_median": 0.03 + index / 100,
                        },
                    }
                }))
            metrics = load_screen_metrics(root)
            self.assertEqual(metrics["appearance"]["psnr"], 19.0)
            self.assertAlmostEqual(metrics["full"]["occupancy"], 0.982)

    def test_load_exact_metrics_preserves_mapping_and_delta_directions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            macro = {}
            records = []
            values = {
                "irregular_seed0": (18.0, 0.80),
                "irregular_seed1": (18.2, 0.75),
                "irregular_seed2": (17.9, 0.82),
            }
            for arm, (psnr, lpips) in values.items():
                macro[arm] = {"psnr": psnr, "lpips": lpips, "ssim": 0.7}
                records.append({
                    "style": "anime",
                    "arm": arm,
                    "lpips_mean": lpips,
                    "render_signature": {"psnr": psnr},
                })
            path.write_text(json.dumps({
                "perceptual_macro": macro,
                "perceptual_records": records,
            }))
            exact, deltas = load_exact_metrics(path)
            self.assertEqual(exact["appearance"]["psnr"], 18.2)
            appearance = next(row for row in deltas if row["arm"] == "appearance")
            full = next(row for row in deltas if row["arm"] == "full")
            self.assertAlmostEqual(appearance["psnr_delta"], 0.2)
            self.assertAlmostEqual(appearance["lpips_improvement"], 0.05)
            self.assertAlmostEqual(full["lpips_improvement"], -0.02)


if __name__ == "__main__":
    unittest.main()
