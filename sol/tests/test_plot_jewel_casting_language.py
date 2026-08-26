"""Tests for casting-language evidence extraction."""

from __future__ import annotations

import unittest

from sol.plot_jewel_casting_language import plot_payload


def _row(size: int) -> dict:
    return {
        "macro": {
            "motif_explained_fraction": size / 1000,
            "token_only_voxel_psnr": 10 + size / 100,
            "half_residual_voxel_psnr": 20 + size / 100,
            "grid_control_voxel_psnr": 5,
            "token_only_cell_center_lock_fraction": 0,
            "grid_control_cell_center_lock_fraction": 1,
        },
        "canonicality": {
            "summary": {
                "cell_conditional_motif_cosine": {
                    "same_source": 0.8,
                    "different_source": 0.6,
                    "margin": 0.2,
                }
            }
        },
    }


class CastingLanguagePlotTests(unittest.TestCase):
    def test_payload_orders_numeric_vocabulary_keys(self) -> None:
        report = {
            "schema": "jewel-casting-language-gate-v0",
            "vocabularies": {"256": _row(256), "64": _row(64)},
            "gate": {"passed": True},
        }
        payload = plot_payload(report)
        self.assertEqual(payload["vocabularies"], [64, 256])
        self.assertTrue(payload["gate_passed"])

    def test_payload_rejects_unknown_schema(self) -> None:
        with self.assertRaises(ValueError):
            plot_payload({"schema": "unknown", "vocabularies": {}})


if __name__ == "__main__":
    unittest.main()
