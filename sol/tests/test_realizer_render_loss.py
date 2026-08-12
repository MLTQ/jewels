"""Differentiable video-to-jewel render-objective tests."""

from __future__ import annotations

import unittest

import torch

from sol.realizer_render_loss import estimate_target_marks, realizer_render_loss
from sol.synthetic import random_jewels


class RealizerRenderLossTests(unittest.TestCase):
    def test_exact_velocity_recovers_target_endpoint(self) -> None:
        noise = torch.randn(5, 22)
        target = torch.randn(5, 22)
        time = torch.tensor([0.35])
        noised = (1 - time) * noise + time * target
        estimate = estimate_target_marks(noised, target - noise, time)
        torch.testing.assert_close(estimate, target)

    def test_render_terms_are_finite_and_backpropagate(self) -> None:
        target = random_jewels(8, seed=71)
        predicted = (target + 0.01).detach().requires_grad_()
        carried = random_jewels(3, seed=72)
        terms = realizer_render_loss(
            predicted,
            target,
            carried,
            total_frames=12,
            frontier=4,
            stride_frames=4,
            background=torch.zeros(3),
            render_height=6,
            render_width=8,
            patches=1,
            patch_frames=2,
            patch_height=2,
            patch_width=2,
            generator=torch.Generator().manual_seed(3),
        )
        terms.total.backward()
        for value in (terms.total, terms.rgb, terms.edge, terms.chroma, terms.structure):
            self.assertTrue(torch.isfinite(value))
        self.assertIsNotNone(predicted.grad)
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_identical_fields_have_zero_objective(self) -> None:
        target = random_jewels(6, seed=73)
        terms = realizer_render_loss(
            target,
            target,
            torch.empty(0, 22),
            total_frames=10,
            frontier=2,
            stride_frames=4,
            background=torch.tensor((0.1, 0.2, 0.3)),
            render_height=4,
            render_width=4,
            patches=1,
            patch_frames=1,
            patch_height=2,
            patch_width=2,
        )
        self.assertLess(float(terms.total.abs()), 1e-6)


if __name__ == "__main__":
    unittest.main()
