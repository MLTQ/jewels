"""Coupled-set checkpoint calibration tests."""

from __future__ import annotations

import unittest

import torch

from sol.calibrate_coupled_set_checkpoint import calibrated_checkpoint


class CalibrateCoupledSetCheckpointTests(unittest.TestCase):
    def test_scales_only_residual_projection_and_records_provenance(self) -> None:
        saved = {
            "model": {
                "base.weight": torch.ones(2, 2),
                "set_blocks.0.row_update.3.weight": torch.full((2, 2), 4.0),
                "set_blocks.0.row_update.3.bias": torch.full((2,), 2.0),
                "set_blocks.0.set_encoder.input.weight": torch.ones(1),
            },
            "optimizer": {"state": "present"},
            "scaler": {"state": "present"},
            "meta": {
                "architecture": "scaffold_birth_mark_flow_v1",
                "model_args": {"set_depth": 1},
            },
        }
        output = calibrated_checkpoint(saved, 0.25, "candidate.pt")
        self.assertTrue(torch.equal(output["model"]["base.weight"], torch.ones(2, 2)))
        self.assertTrue(
            torch.equal(
                output["model"]["set_blocks.0.row_update.3.weight"],
                torch.ones(2, 2),
            )
        )
        self.assertIsNone(output["optimizer"])
        self.assertEqual(output["meta"]["coupled_set_calibration"]["strength"], 0.25)
        self.assertNotIn("coupled_set_calibration", saved["meta"])

    def test_rejects_second_calibration(self) -> None:
        saved = {
            "model": {
                "set_blocks.0.row_update.3.weight": torch.ones(1),
                "set_blocks.0.row_update.3.bias": torch.ones(1),
            },
            "meta": {
                "architecture": "scaffold_birth_mark_flow_v1",
                "model_args": {"set_depth": 1},
                "coupled_set_calibration": {"strength": 0.5},
            },
        }
        with self.assertRaisesRegex(ValueError, "already"):
            calibrated_checkpoint(saved, 0.5, "candidate.pt")


if __name__ == "__main__":
    unittest.main()
