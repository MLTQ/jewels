"""Tests for read-only responsibility calibration helpers."""

from __future__ import annotations

import unittest

import torch

from sol.calibrate_responsibility_distillation import parameter_gradient_norm


class ResponsibilityCalibrationTests(unittest.TestCase):
    def test_parameter_gradient_norm_respects_group_and_unused_parameters(self) -> None:
        used = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        unused = torch.nn.Parameter(torch.tensor([8.0]))
        loss = used.sum()
        self.assertAlmostEqual(
            parameter_gradient_norm(loss, [used, unused], retain_graph=False),
            2 ** 0.5,
        )


if __name__ == "__main__":
    unittest.main()
