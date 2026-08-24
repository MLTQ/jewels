"""Tests for appearance-contract evidence parsing."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sol.plot_appearance_contract_evidence import load_evidence


class AppearanceContractEvidenceTests(unittest.TestCase):
    def test_load_evidence_preserves_arm_and_delta_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bounded_screens = root / "bounded"
            residual_screens = root / "residual"
            for screen_root, names in (
                (bounded_screens, ("control", "hybrid_midpoint")),
                (residual_screens, ("control", "response")),
            ):
                for index, name in enumerate(names):
                    run = screen_root / f"{name}_seed0_600"
                    run.mkdir(parents=True)
                    (run / "summary.json").write_text(json.dumps({
                        "latest_evaluation": {
                            "macro_psnr": 18.0 + index,
                            "structure": {
                                "occupancy_uniformity": 0.98,
                                "active_fraction": 0.63,
                                "mixed_spacetime_tilt_median": 0.52,
                            },
                        }
                    }))

            def report(first: tuple[float, float], second: tuple[float, float]) -> dict:
                macro, records = {}, []
                for arm, values in zip(
                    ("irregular_seed0", "irregular_seed1"), (first, second)
                ):
                    psnr, lpips = values
                    macro[arm] = {
                        "psnr": psnr, "lpips": lpips, "ssim": 0.8,
                        "layout_psnr": 21.0,
                    }
                    records.append({
                        "style": "anime", "arm": arm, "lpips_mean": lpips,
                        "render_signature": {"psnr": psnr},
                    })
                return {"perceptual_macro": macro, "perceptual_records": records}

            bounded_audit = root / "bounded.json"
            residual_audit = root / "residual.json"
            bounded_audit.write_text(json.dumps(report((18.0, 0.8), (17.9, 0.75))))
            residual_audit.write_text(json.dumps(report((19.0, 0.7), (19.2, 0.68))))
            metrics, deltas = load_evidence(
                bounded_screens, bounded_audit, residual_screens, residual_audit
            )
            self.assertEqual(metrics["bounded_midpoint"]["psnr"], 17.9)
            self.assertEqual(metrics["raw_response"]["psnr"], 19.2)
            self.assertAlmostEqual(deltas[0]["psnr_delta"], 0.2)
            self.assertAlmostEqual(deltas[0]["lpips_improvement"], 0.02)


if __name__ == "__main__":
    unittest.main()
