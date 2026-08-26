"""Tests for hierarchical-language evidence extraction."""

from __future__ import annotations

import unittest

from sol.plot_hierarchical_jewel_casting_language import hierarchical_plot_payload


class HierarchicalPlotTests(unittest.TestCase):
    def test_fresh_result_is_appended_after_granularity_controls(self) -> None:
        granularity = {
            "schema": "jewel-casting-granularity-gate-v1",
            "curve": [
                {
                    "bundle_size": 2,
                    "token_only_voxel_psnr": 18,
                    "half_residual_voxel_psnr": 22,
                    "half_residual_mixed_tilt_retention": 1,
                    "canonical_margin": 0.1,
                    "eight_frame_discrete_decisions": 23000,
                }
            ],
        }
        hierarchy = {
            "schema": "hierarchical-jewel-casting-language-gate-v1",
            "macro": {
                "token_only_voxel_psnr": 21,
                "half_residual_voxel_psnr": 26,
                "half_residual_mixed_tilt_retention": 1,
                "eight_frame_decisions": 35000,
            },
            "canonicality": {"summary": {"margin": 0.2}},
            "gate": {"passed": True},
        }
        payload = hierarchical_plot_payload(hierarchy, granularity)
        self.assertEqual(payload["labels"], ["bundle 2", "hierarchy\n(fresh)"])
        self.assertEqual(payload["token_psnr"], [18, 21])
        self.assertTrue(payload["gate_passed"])


if __name__ == "__main__":
    unittest.main()
