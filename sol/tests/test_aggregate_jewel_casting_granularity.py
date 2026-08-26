"""Tests for Jewel casting granularity gate aggregation."""

from __future__ import annotations

import unittest

from sol.aggregate_jewel_casting_granularity import evaluate_granularity_gate


def _report(bundle_size: int, token_psnr: float, half_psnr: float) -> dict:
    decisions = 72000 / bundle_size * 4
    return {
        "schema": "factorized-jewel-casting-language-gate-v1",
        "protocol": {"bundle_size": bundle_size},
        "vocabularies": {
            "1024": {
                "macro": {
                    "casts": 72000 / bundle_size,
                    "discrete_decisions": decisions,
                    "jewels_per_cast": bundle_size,
                    "motif_explained_fraction": 0.5,
                    "factor_explained_fraction": {
                        "layout": 0.5, "covariance": 0.5,
                        "surface": 0.5, "gradient": 0.5,
                    },
                    "token_only_voxel_psnr": token_psnr,
                    "half_residual_voxel_psnr": half_psnr,
                    "full_residual_voxel_psnr": 120,
                    "token_only_mixed_tilt_retention": 1,
                    "half_residual_mixed_tilt_retention": 1,
                    "token_only_cell_center_lock_fraction": 0,
                    "grid_control_voxel_psnr": 6,
                    "grid_control_cell_center_lock_fraction": 1,
                },
                "canonicality": {
                    "summary": {
                        "composite_cell_conditional_cosine": {"margin": 0.1}
                    }
                },
                "records": [
                    {
                        "source_jewels": 72000,
                        "program": {"source_jewels": 72000, "serialized_jewels": 72000},
                    }
                ],
            }
        },
    }


class CastingGranularityTests(unittest.TestCase):
    def test_registered_upper_bound_passes_clean_curve(self) -> None:
        reports = {
            8: _report(8, 15, 17),
            4: _report(4, 17, 20),
            2: _report(2, 19, 23),
            1: _report(1, 21, 26),
        }
        result = evaluate_granularity_gate(reports)
        self.assertTrue(result["gate"]["passed"])

    def test_requires_every_registered_bundle_size(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_granularity_gate({8: _report(8, 15, 17)})


if __name__ == "__main__":
    unittest.main()
