"""Tests for label-free foreground, motion-boundary, and quiet-region metrics."""

from __future__ import annotations

import unittest

import torch

from sol.saliency_metrics import saliency_render_signature


class SaliencyMetricTests(unittest.TestCase):
    def test_identical_videos_have_zero_error(self) -> None:
        target = torch.rand(4, 5, 6, 3)
        signature = saliency_render_signature(
            target, target, background=torch.zeros(3)
        )
        self.assertEqual(signature.foreground_psnr, 100.0)
        for value in (
            signature.foreground_rgb_mae,
            signature.foreground_edge_mae,
            signature.motion_boundary_mae,
            signature.quiet_temporal_mae,
        ):
            self.assertEqual(value, 0.0)

    def test_missing_moving_color_has_saliency_error(self) -> None:
        target = torch.zeros(4, 5, 6, 3)
        target[1, 2, 2, 0] = 1
        target[2, 2, 3, 0] = 1
        signature = saliency_render_signature(
            torch.zeros_like(target), target, background=torch.zeros(3)
        )
        self.assertGreater(signature.foreground_rgb_mae, 0)
        self.assertGreater(signature.foreground_edge_mae, 0)
        self.assertGreater(signature.motion_boundary_mae, 0)

    def test_static_target_exposes_candidate_flicker(self) -> None:
        target = torch.full((4, 5, 6, 3), 0.5)
        candidate = target.clone()
        candidate[1::2] += 0.1
        signature = saliency_render_signature(
            candidate, target, background=torch.zeros(3)
        )
        self.assertGreater(signature.quiet_temporal_mae, 0)


if __name__ == "__main__":
    unittest.main()
