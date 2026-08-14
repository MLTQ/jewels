"""Tests for perceptual arm scoring with an injected metric."""

from __future__ import annotations

import unittest

import torch

from sol.perceptual_eval import score_arms


class PerceptualEvalTests(unittest.TestCase):
    def test_score_arms_reports_per_frame_and_mean(self) -> None:
        target = torch.rand(4, 6, 8, 3)
        arms = {
            "identical": target.clone(),
            "noisy": (target + 0.1).clamp(0, 1),
        }

        def metric(candidate: torch.Tensor, reference: torch.Tensor) -> list[float]:
            return [
                float((candidate[index] - reference[index]).abs().mean())
                for index in range(len(candidate))
            ]

        report = score_arms(target, arms, metric)
        self.assertEqual(len(report["identical"]["lpips_per_frame"]), 4)
        self.assertEqual(report["identical"]["lpips_mean"], 0.0)
        self.assertGreater(report["noisy"]["lpips_mean"], 0.0)
        self.assertIn("psnr", report["noisy"]["render_signature"])

    def test_score_arms_rejects_shape_mismatch(self) -> None:
        target = torch.rand(4, 6, 8, 3)
        with self.assertRaisesRegex(ValueError, "does not match"):
            score_arms(
                target,
                {"bad": torch.rand(4, 6, 9, 3)},
                lambda a, b: [0.0] * len(a),
            )


if __name__ == "__main__":
    unittest.main()
