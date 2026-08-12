"""Tests for differentiable frontier visible-contribution supervision."""

from __future__ import annotations

import unittest

import torch

from sol.frontier_contribution_loss import (
    frontier_contribution_loss,
    frontier_peak_alpha,
)


def _features() -> torch.Tensor:
    features = torch.zeros(4, 22)
    features[:, 3] = -5.0
    features[:, 6] = -5.0
    features[:, 8] = -3.0
    features[:, 21] = 1.0
    return features


class FrontierContributionLossTests(unittest.TestCase):
    def test_frontier_peak_alpha_rewards_temporal_alignment(self) -> None:
        aligned = _features()
        shifted = aligned.clone()
        shifted[:, 2] = 0.5
        self.assertTrue((frontier_peak_alpha(aligned) > frontier_peak_alpha(shifted)).all())

    def test_identical_marks_are_zero_and_shifted_marks_backpropagate(self) -> None:
        target = _features()
        identical = frontier_contribution_loss(
            target, target, torch.tensor([0, 0, 1, 1]), n_cells=2
        )
        self.assertLess(float(identical.total.abs()), 1e-7)

        predicted = target.clone()
        predicted[:, 2] = 0.4
        predicted.requires_grad_()
        terms = frontier_contribution_loss(
            predicted, target, torch.tensor([0, 0, 1, 1]), n_cells=2
        )
        terms.total.backward()
        self.assertGreater(float(terms.total.detach()), 0)
        self.assertTrue(torch.isfinite(predicted.grad).all())


if __name__ == "__main__":
    unittest.main()
