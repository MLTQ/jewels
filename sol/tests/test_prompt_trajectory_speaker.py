"""Tests for prompt-only trajectory-token compilation."""

from __future__ import annotations

import unittest

from sol.prompt_trajectory_speaker import PromptTrajectorySpeaker


class PromptTrajectorySpeakerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.speaker = PromptTrajectorySpeaker(
            prompts=("dancer", "dog", "welder"),
            scene_sources=((0, 1, 2), (3, 4, 5), (6, 7, 8)),
        )

    def test_prompt_seed_is_deterministic_and_distinct(self) -> None:
        first = self.speaker.compile("dog", 41)
        second = self.speaker.compile("dog", 41)
        self.assertEqual(first, second)
        self.assertEqual(first.scene_token, 1)
        self.assertNotEqual(first.foreground_token, first.background_token)
        self.assertIn(first.foreground_token, (3, 4, 5))

    def test_controls_change_scene_without_target_input(self) -> None:
        shuffled = self.speaker.compile_shuffled("dog", 41)
        null = self.speaker.compile_null(41)
        self.assertEqual(shuffled.scene_token, 2)
        self.assertEqual(null.scene_token, 2)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.speaker.compile("cat", 41)


if __name__ == "__main__":
    unittest.main()
