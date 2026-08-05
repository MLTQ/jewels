"""Tests for continuation visualization point-grid construction."""

from __future__ import annotations

import unittest

import torch

from sol.render_streaming_continuation import frame_points
from sol.streaming import frame_times


class RenderStreamingContinuationTests(unittest.TestCase):
    def test_frame_points_use_global_time_and_full_spatial_extent(self) -> None:
        indices = torch.tensor([2, 5])
        points = frame_points(8, indices, 3, 4, device=torch.device("cpu"))
        self.assertEqual(points.shape, (24, 3))
        reshaped = points.reshape(2, 3, 4, 3)
        self.assertTrue(torch.allclose(reshaped[:, 0, 0, 2], frame_times(8)[indices]))
        self.assertEqual(float(reshaped[0, 0, 0, 0]), -1.0)
        self.assertEqual(float(reshaped[0, -1, -1, 0]), 1.0)
        self.assertEqual(float(reshaped[0, 0, 0, 1]), -1.0)
        self.assertEqual(float(reshaped[0, -1, -1, 1]), 1.0)

if __name__ == "__main__":
    unittest.main()
