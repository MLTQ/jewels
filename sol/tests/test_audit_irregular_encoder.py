"""Tests for irregular encoder audit structure metrics."""

from __future__ import annotations

import unittest

import torch

from sol.audit_irregular_encoder import layout_slice, structure, summarize_gate


class IrregularAuditTests(unittest.TestCase):
    def test_layout_slice_filters_opacity_plane_and_bounds_count(self) -> None:
        features = torch.zeros(12, 22)
        features[:, 0] = torch.linspace(-1, 1, 12)
        features[:, 2] = 0.05
        features[:, 21] = 0.0
        features[0, 21] = -10.0
        features[1, 2] = 0.5
        selected = layout_slice(features, fixed_axis=2, band=0.12, max_points=4)
        self.assertEqual(selected.shape, (4, 3))
        self.assertTrue(torch.all(selected[:, 2].abs() <= 0.12))

    def test_structure_reports_active_fraction_and_mixed_tilt(self) -> None:
        features = torch.zeros(20, 22)
        features[:, :3] = torch.rand(20, 3) * 2 - 1
        features[:, 3] = -4.0
        features[:, 6] = -2.0
        features[:, 8] = -2.0
        features[:10, 21] = 0.0
        features[10:, 21] = -10.0
        report = structure(features)
        self.assertEqual(report["jewels_total"], 20)
        self.assertAlmostEqual(report["active_fraction"], 0.5)
        self.assertGreaterEqual(report["mixed_spacetime_tilt_median"], 0.0)
        self.assertLessEqual(report["mixed_spacetime_tilt_median"], 1.0)

    def test_preregistered_gate_requires_every_seed_to_degrid(self) -> None:
        perceptual = {
            f"irregular_seed{seed}": {
                "lpips": 0.3,
                "psnr": 21.0,
                "ssim": 0.7,
                "layout_psnr": 22.0,
                "layout_ssim": 0.8,
            }
            for seed in range(3)
        }
        structure_macro = {
            "lattice": {"occupancy_uniformity": 0.999, "active_fraction": 1.0},
            "irregular": {
                "occupancy_uniformity": 0.97,
                "active_fraction": 0.6,
                "mixed_spacetime_tilt_median": 0.3,
            },
        }
        by_seed = {
            f"seed{seed}": {"occupancy_uniformity": 0.97}
            for seed in range(3)
        }
        self.assertTrue(summarize_gate(perceptual, structure_macro, by_seed)["passed"])
        by_seed["seed2"]["occupancy_uniformity"] = 0.99
        self.assertFalse(summarize_gate(perceptual, structure_macro, by_seed)["passed"])


if __name__ == "__main__":
    unittest.main()
