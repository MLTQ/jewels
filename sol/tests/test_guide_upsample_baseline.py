"""Tests for the trivial guide-decode baseline."""

from __future__ import annotations

import unittest

import torch

from sol.guide_upsample_baseline import (
    cell_raster_to_video,
    evaluate_source,
    guide_upsample_baseline,
)
from sol.token_grid import GridSpec
from sol.video_guide import video_to_cell_raster


class GuideUpsampleBaselineTests(unittest.TestCase):
    def test_constant_video_round_trips_exactly(self) -> None:
        spec = GridSpec((4, 4, 2), 1)
        video = torch.full((8, 12, 20, 3), 0.25)
        baseline = guide_upsample_baseline(video, spec, stride_frames=4, strides=2)
        self.assertEqual(baseline.shape, (8, 12, 20, 3))
        self.assertTrue(torch.allclose(baseline, video, atol=1e-6))

    def test_inverse_preserves_axis_order(self) -> None:
        spec = GridSpec((4, 4, 2), 1)
        video = torch.zeros(4, 16, 16, 3)
        video[:2, :4, 12:, 0] = 1.0
        guide = video_to_cell_raster(video, spec)
        decoded = cell_raster_to_video(guide, spec, 4, 16, 16)
        bright = (decoded[..., 0] > decoded[..., 0].mean()).nonzero()
        self.assertLess(bright[:, 0].float().mean(), 2.0)
        self.assertLess(bright[:, 1].float().mean(), 8.0)
        self.assertGreater(bright[:, 2].float().mean(), 8.0)

    def test_rejects_incomplete_strides(self) -> None:
        spec = GridSpec((4, 4, 2), 1)
        video = torch.rand(7, 8, 8, 3)
        with self.assertRaisesRegex(ValueError, "complete strides"):
            guide_upsample_baseline(video, spec, stride_frames=4, strides=2)

    def test_evaluate_source_matches_rollout_reference_slice(self) -> None:
        spec = GridSpec((4, 4, 2), 1)
        video = torch.rand(9, 8, 8, 3)
        report = evaluate_source(
            video,
            spec,
            stride_frames=4,
            strides=2,
            background=torch.tensor([0.5, 0.5, 0.5]),
        )
        self.assertEqual(report["completed_frames"], 8)
        self.assertGreater(report["render_signature"]["psnr"], 0.0)
        self.assertIn("quiet_temporal_mae", report["saliency_signature"])
        self.assertIn("seam_to_regular_ratio", report["seams"])


if __name__ == "__main__":
    unittest.main()
