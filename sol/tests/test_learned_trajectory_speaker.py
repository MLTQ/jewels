"""Tests for the learned trajectory-token speaker."""

from __future__ import annotations

import unittest

import torch

from sol.learned_trajectory_speaker import LearnedTrajectorySpeaker, trajectory_program_loss


class LearnedTrajectorySpeakerTests(unittest.TestCase):
    def test_forward_loss_and_sampling(self) -> None:
        torch.manual_seed(5)
        model = LearnedTrajectorySpeaker(8, 16, 3, 9)
        text = torch.randn(4, 8)
        scene = torch.tensor([0, 1, 2, 0])
        foreground = torch.tensor([0, 3, 6, 1])
        background = torch.tensor([1, 4, 7, 2])
        predictions = model(text, scene, foreground)
        self.assertEqual(predictions["scene_logits"].shape, (4, 3))
        self.assertEqual(predictions["foreground_logits"].shape, (4, 9))
        self.assertEqual(predictions["background_logits"].shape, (4, 9))
        loss, parts = trajectory_program_loss(
            predictions, scene, foreground, background
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(parts["total"], 0)
        first = model.sample(
            text[:1], generator=torch.Generator().manual_seed(7)
        )
        second = model.sample(
            text[:1], generator=torch.Generator().manual_seed(7)
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first.foreground_token, first.background_token)
        with self.assertRaisesRegex(ValueError, "one text"):
            model.sample(text, generator=torch.Generator())


if __name__ == "__main__":
    unittest.main()
