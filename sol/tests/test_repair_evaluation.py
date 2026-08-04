"""Tests for cuboid-mask sampling and repair evaluation metrics."""

from __future__ import annotations

import unittest

import torch

from sol.repair_evaluation import evaluate_masked_repair, sample_cuboid_masks


def _zero_velocity(
    state: torch.Tensor,
    _time: torch.Tensor,
    _condition: torch.Tensor | None,
) -> torch.Tensor:
    return torch.zeros_like(state)


class RepairEvaluationTests(unittest.TestCase):
    def test_fixed_extent_masks_have_expected_volume(self) -> None:
        masks = sample_cuboid_masks(
            5,
            (6, 5, 4),
            (2, 3, 4),
            (2, 3, 4),
            device="cpu",
            generator=torch.Generator().manual_seed(9),
        )
        self.assertEqual(masks.shape, (5, 120))
        self.assertTrue(bool((masks.sum(dim=1) == 24).all()))

    def test_evaluation_reports_exact_clean_clamp(self) -> None:
        latents = torch.randn(3, 8, 4)
        result = evaluate_masked_repair(
            _zero_velocity,
            latents,
            torch.randn(3, 5),
            (2, 2, 2),
            (1, 1, 1),
            (1, 1, 1),
            device="cpu",
            examples=3,
            steps=2,
            seed=2,
        )
        self.assertEqual(result.clean_max_abs_error, 0.0)
        self.assertEqual(result.examples, 3)
        self.assertAlmostEqual(result.mean_dirty_fraction, 0.125)


if __name__ == "__main__":
    unittest.main()
