"""Video-to-jewel guide alignment tests."""

from __future__ import annotations

import unittest

import torch

from sol.token_grid import GridSpec
from sol.video_guide import video_to_cell_raster


class VideoGuideTests(unittest.TestCase):
    def test_preserves_axis_order_at_matching_resolution(self) -> None:
        spec = GridSpec((3, 2, 2), 4)
        video = torch.zeros(2, 2, 3, 3)
        for t in range(2):
            for v in range(2):
                for u in range(3):
                    video[t, v, u] = torch.tensor((u, v, t))
        guide = video_to_cell_raster(video, spec)
        for u in range(3):
            for v in range(2):
                for t in range(2):
                    cell = (u * 2 + v) * 2 + t
                    self.assertTrue(
                        torch.equal(guide[cell], torch.tensor((u, v, t)).float())
                    )


if __name__ == "__main__":
    unittest.main()
