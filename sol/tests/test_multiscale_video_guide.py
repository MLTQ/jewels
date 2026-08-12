"""Multiscale video-guide alignment and feature tests."""

from __future__ import annotations

import unittest

import torch

from sol.multiscale_video_guide import (
    MULTISCALE_GUIDE_FEATURE_DIM,
    video_to_multiscale_cell_tokens,
)
from sol.token_grid import GridSpec


class MultiscaleVideoGuideTests(unittest.TestCase):
    def test_matching_subgrid_preserves_within_cell_samples(self) -> None:
        spec = GridSpec((1, 1, 1), 8)
        video = torch.zeros(2, 2, 2, 3)
        for t in range(2):
            for v in range(2):
                for u in range(2):
                    video[t, v, u] = torch.tensor((u, v, t))
        tokens = video_to_multiscale_cell_tokens(
            video, spec, scales=(1,), subgrid=(2, 2, 2)
        )
        self.assertEqual(tokens.shape, (1, 8, MULTISCALE_GUIDE_FEATURE_DIM))
        expected = []
        for u in range(2):
            for v in range(2):
                for t in range(2):
                    expected.append((u, v, t))
        torch.testing.assert_close(tokens[0, :, :3], torch.tensor(expected).float())

    def test_cells_follow_u_v_t_flatten_order(self) -> None:
        spec = GridSpec((3, 2, 2), 4)
        video = torch.zeros(2, 2, 3, 3)
        for t in range(2):
            for v in range(2):
                for u in range(3):
                    video[t, v, u] = torch.tensor((u, v, t))
        tokens = video_to_multiscale_cell_tokens(
            video, spec, scales=(1,), subgrid=(1, 1, 1)
        )
        for u in range(3):
            for v in range(2):
                for t in range(2):
                    cell = (u * 2 + v) * 2 + t
                    torch.testing.assert_close(
                        tokens[cell, 0, :3], torch.tensor((u, v, t)).float()
                    )

    def test_multiscale_tokens_include_motion_derivatives(self) -> None:
        spec = GridSpec((2, 2, 2), 4)
        video = torch.zeros(4, 4, 4, 3)
        video[2:] = 1
        tokens = video_to_multiscale_cell_tokens(video, spec)
        self.assertEqual(tokens.shape, (spec.n_cells, 24, 16))
        self.assertTrue(torch.isfinite(tokens).all())
        self.assertGreater(float(tokens[..., 9:12].abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
