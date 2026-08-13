"""Differentiable video-to-jewel render-objective tests."""

from __future__ import annotations

import unittest

import torch

from sol.realizer_render_loss import (
    _sample_patch_points,
    estimate_target_marks,
    realizer_render_loss,
    scaffold_saliency_weights,
)
from sol.synthetic import random_jewels


class RealizerRenderLossTests(unittest.TestCase):
    def test_importance_sampling_addresses_the_selected_cell(self) -> None:
        importance = torch.zeros(8)
        importance[7] = 1
        _, local = _sample_patch_points(
            total_frames=8,
            frontier=0,
            stride_frames=4,
            render_height=4,
            render_width=4,
            patches=2,
            patch_frames=1,
            patch_height=1,
            patch_width=1,
            anchor_frontier=False,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(8),
            patch_importance=importance,
            importance_grid_shape=(2, 2, 2),
            importance_fraction=1,
        )
        torch.testing.assert_close(local, torch.full((2, 3), 0.75))

    def test_scaffold_saliency_prefers_motion_and_chroma(self) -> None:
        guide = torch.zeros(8, 3)
        guide[1] = torch.tensor((1.0, 0.0, 0.0))
        weights = scaffold_saliency_weights(
            guide, (2, 2, 2), torch.zeros(3)
        )
        self.assertGreater(float(weights[1]), float(weights[6]))

    def test_frontier_anchor_places_first_patch_at_local_zero(self) -> None:
        _, local = _sample_patch_points(
            total_frames=12,
            frontier=4,
            stride_frames=4,
            render_height=6,
            render_width=8,
            patches=2,
            patch_frames=2,
            patch_height=2,
            patch_width=2,
            anchor_frontier=True,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(8),
        )
        first_patch_points = 2 * 2 * 2
        self.assertEqual(float(local[:first_patch_points, 2].min()), 0.0)

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
            guide_raster=torch.rand(8, 3),
            guide_grid_shape=(2, 2, 2),
            saliency_fraction=1,
            saliency_weight=0.5,
            motion_weight=0.5,
            stability_weight=0.5,
            anchor_frontier=True,
            generator=torch.Generator().manual_seed(3),
        )
        terms.total.backward()
        for value in (
            terms.total,
            terms.rgb,
            terms.edge,
            terms.chroma,
            terms.structure,
            terms.saliency_rgb,
            terms.motion,
            terms.stability,
        ):
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

    def test_candidate_background_receives_visual_gradient(self) -> None:
        target = random_jewels(6, seed=74)
        candidate_background = torch.full((3,), 0.2, requires_grad=True)
        terms = realizer_render_loss(
            target,
            target,
            torch.empty(0, 22),
            total_frames=10,
            frontier=2,
            stride_frames=4,
            background=torch.tensor((0.4, 0.5, 0.6)),
            candidate_background=candidate_background,
            render_height=4,
            render_width=4,
            patches=1,
            patch_frames=1,
            patch_height=2,
            patch_width=2,
        )
        terms.total.backward()
        self.assertIsNotNone(candidate_background.grad)
        self.assertGreater(float(candidate_background.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
