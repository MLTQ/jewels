"""Coordinate-layout test for prior comparison rendering."""

from __future__ import annotations

import unittest

import torch

from sol.render_prior_samples import frame_points


class PriorRenderTests(unittest.TestCase):
    def test_selected_frame_points_match_normalized_video_order(self) -> None:
        points = frame_points(
            (5, 2, 3), torch.tensor([0, 4]), device=torch.device("cpu")
        ).reshape(2, 2, 3, 3)
        torch.testing.assert_close(points[0, 0, 0], torch.tensor([-1.0, -1.0, -1.0]))
        torch.testing.assert_close(points[1, -1, -1], torch.tensor([1.0, 1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
