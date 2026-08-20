"""Tests for support-correct scaling experiment metrics."""

from __future__ import annotations

import unittest

import torch

from sol.support_correct_scaling import field_structure, summarize
from stprim.core.params import PrimitiveField


class SupportCorrectScalingTests(unittest.TestCase):
    def test_field_structure_detects_temporal_tube(self) -> None:
        field = PrimitiveField(2, p1_color=False)
        with torch.no_grad():
            field.log_scale[:] = torch.log(torch.tensor([0.1, 0.1, 0.5]))
            field.quat.zero_()
            field.quat[:, 0] = 1.0
        report = field_structure(field, frames=21, t_scale=1.0)
        self.assertAlmostEqual(report["anisotropy_median"], 5.0, places=5)
        self.assertAlmostEqual(
            report["principal_temporal_alignment_median"], 1.0, places=5
        )
        self.assertAlmostEqual(report["mixed_spacetime_tilt_median"], 0.0, places=5)
        self.assertAlmostEqual(
            report["five_sigma_lifespan_frames_median"], 50.0, places=4
        )

    def test_summary_uses_support_evaluation_curve(self) -> None:
        base = {
            "cull_mode": "support",
            "fit_seconds": 1.0,
            "renderer_gap_max_abs": 0.0,
            "structure": {
                "anisotropy_median": 2.0,
                "principal_temporal_alignment_p90": 0.8,
            },
        }
        records = [
            {**base, "steps": 10, "support_eval_psnr_db": 20.0},
            {**base, "steps": 20, "fit_seconds": 2.0, "support_eval_psnr_db": 26.0},
        ]
        report = summarize(records)
        self.assertEqual(report["support"]["support_psnr_gain_db"], 6.0)
        self.assertTrue(report["initial_proof_gate"]["positive_compute_slope"])
        self.assertTrue(
            report["initial_proof_gate"]["largest_support_run_reaches_25db"]
        )


if __name__ == "__main__":
    unittest.main()
