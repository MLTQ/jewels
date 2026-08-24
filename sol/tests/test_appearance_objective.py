"""Tests for frozen-field full-frame appearance objectives."""

from __future__ import annotations

import unittest

import torch

from sol.appearance_objective import (
    appearance_objective,
    multiscale_image_loss,
    range_diagnostics,
    residual_energy,
    temporal_edge_loss,
)


class AppearanceObjectiveTests(unittest.TestCase):
    def test_legacy_wrapper_matches_registered_default(self) -> None:
        rendered = torch.rand(3, 12, 16, 3)
        target = torch.rand_like(rendered)
        self.assertEqual(
            float(multiscale_image_loss(rendered, target)),
            float(appearance_objective(rendered, target).total),
        )

    def test_matching_video_has_only_charbonnier_floor(self) -> None:
        video = torch.rand(3, 12, 16, 3)
        terms = appearance_objective(
            video, video, temporal_weight=0.5, structure_weight=0.25,
        )
        self.assertLess(float(terms.total), 0.0011)
        self.assertEqual(float(terms.temporal), 0.0)
        self.assertLess(float(terms.structure.abs()), 1e-6)

    def test_temporal_error_backpropagates(self) -> None:
        target = torch.zeros(3, 8, 8, 3)
        rendered = target.clone()
        rendered[1] = 1
        rendered.requires_grad_()
        value = temporal_edge_loss(rendered, target)
        self.assertGreater(float(value.detach()), 0.5)
        value.backward()
        self.assertGreater(float(rendered.grad.abs().sum()), 0.0)

    def test_range_and_residual_diagnostics_separate_failure_modes(self) -> None:
        values = torch.tensor((-0.5, 0.5, 1.5))
        diagnostics = range_diagnostics(values)
        self.assertAlmostEqual(float(diagnostics["out_of_range_fraction"]), 2 / 3)
        self.assertGreater(float(diagnostics["range_excess"]), 0.0)
        residual = torch.zeros(2, 12)
        residual[:, :3] = 2
        color, gradient = residual_energy(residual)
        self.assertEqual(float(color), 4.0)
        self.assertEqual(float(gradient), 0.0)

    def test_negative_or_empty_weight_contract_is_rejected(self) -> None:
        video = torch.zeros(1, 4, 4, 3)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            appearance_objective(video, video, rgb_weight=-1)
        with self.assertRaisesRegex(ValueError, "not all zero"):
            appearance_objective(video, video, rgb_weight=0, spatial_weight=0)


if __name__ == "__main__":
    unittest.main()
