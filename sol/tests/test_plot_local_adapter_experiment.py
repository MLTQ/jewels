"""Tests for local-adapter evidence collection."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sol.plot_local_adapter_experiment import collect_evidence


class LocalAdapterPlotTests(unittest.TestCase):
    def test_collect_evidence_preserves_semantic_arm_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = {
                "audit_final_seed0_400": (0.72, 20.08),
                "audit_seed0_400": (0.73, 20.00),
                "audit_lpips_strength_seed0_400": (0.71, 20.10),
                "audit_derivative_seed0_400": (0.725, 20.05),
            }
            for name, (lpips, psnr) in reports.items():
                (root / name).mkdir()
                macro = {
                    f"irregular_seed{index}": {
                        "lpips": lpips - 0.001 * index,
                        "psnr": psnr + 0.01 * index,
                    }
                    for index in range(4)
                }
                (root / name / "report.json").write_text(json.dumps({
                    "perceptual_macro": macro
                }))
            for name in (
                "radius2_render_seed0_400", "radius2_lpips001_seed0_400",
                "radius2_lpips005_seed0_400",
                "derivative_scale32_lpips005_seed0_400",
            ):
                path = root / "screens" / name
                path.mkdir(parents=True)
                (path / "train_log.jsonl").write_text(json.dumps({
                    "render_out_of_range_fraction": 0.03,
                    "residual_gradient_energy": 0.05,
                }) + "\n")
            evidence = collect_evidence(root)
            self.assertEqual(
                evidence["arms"]["frozen source"]["lpips"], 0.72
            )
            self.assertIn("LPIPS .01", evidence["causal_radius2_minus_radius0"])
            self.assertEqual(
                evidence["diagnostics"]["raw LPIPS .05"]["out_of_range_percent"],
                3.0,
            )


if __name__ == "__main__":
    unittest.main()
