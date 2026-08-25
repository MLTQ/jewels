"""Tests for differentiable perceptual appearance supervision."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from sol.perceptual_objective import perceptual_training_loss


class _Distance(nn.Module):
    def forward(
        self, candidate: torch.Tensor, target: torch.Tensor, *, normalize: bool
    ) -> torch.Tensor:
        if not normalize:
            raise AssertionError("LPIPS-compatible normalization must be requested")
        return (candidate - target).square().mean(dim=(1, 2, 3), keepdim=True)


class PerceptualObjectiveTests(unittest.TestCase):
    def test_loss_is_differentiable_to_render_but_not_metric(self) -> None:
        candidate = torch.full((2, 16, 20, 3), 0.25, requires_grad=True)
        target = torch.full_like(candidate, 0.75)
        metric = _Distance()
        loss = perceptual_training_loss(candidate, target, metric)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertGreater(float(candidate.grad.abs().sum()), 0.0)
        self.assertEqual(list(metric.parameters()), [])

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "share shape"):
            perceptual_training_loss(
                torch.zeros(1, 8, 8, 3),
                torch.zeros(1, 8, 9, 3),
                _Distance(),
            )


if __name__ == "__main__":
    unittest.main()
