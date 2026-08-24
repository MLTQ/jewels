"""Tests for read-only frozen-appearance gradient calibration."""

from __future__ import annotations

import unittest

import torch

from sol.calibrate_frozen_appearance import gradient_l2_norm


class FrozenAppearanceCalibrationTests(unittest.TestCase):
    def test_gradient_norm_does_not_accumulate_parameter_gradients(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor((3.0, 4.0)))
        loss = parameter.square().sum()
        norm = gradient_l2_norm(loss, [parameter], retain_graph=False)
        self.assertAlmostEqual(norm, 10.0)
        self.assertIsNone(parameter.grad)

    def test_unused_parameters_contribute_zero(self) -> None:
        used = torch.nn.Parameter(torch.tensor(2.0))
        unused = torch.nn.Parameter(torch.tensor(7.0))
        norm = gradient_l2_norm(used.square(), [used, unused], retain_graph=False)
        self.assertEqual(norm, 4.0)
        self.assertIsNone(unused.grad)


if __name__ == "__main__":
    unittest.main()
