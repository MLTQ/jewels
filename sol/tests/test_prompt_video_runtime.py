"""Tests for prompt-video naming, validation, and encoding contracts."""

from __future__ import annotations

from pathlib import Path
import unittest

from sol.prompt_video_runtime import (
    ffmpeg_command,
    normalize_prompt,
    realization_seed,
    video_basename,
)


class PromptVideoRuntimeTests(unittest.TestCase):
    def test_prompt_normalization_and_limit(self) -> None:
        self.assertEqual(normalize_prompt("  a  dancing\n dog "), "a dancing dog")
        with self.assertRaisesRegex(ValueError, "enter a prompt"):
            normalize_prompt(" \n ")
        with self.assertRaisesRegex(ValueError, "300"):
            normalize_prompt("x" * 301)

    def test_video_names_are_stable_and_condition_owned(self) -> None:
        first = video_basename("a dancing dog", "exact", 41)
        self.assertEqual(first, video_basename(" a  dancing dog ", "exact", 41))
        self.assertNotEqual(first, video_basename("a dancing dog", "learned", 41))
        self.assertNotEqual(first, video_basename("a dancing dog", "exact", 42))
        self.assertNotIn(" ", first)
        with self.assertRaisesRegex(ValueError, "mode"):
            video_basename("a dancing dog", "../../escape", 42)

    def test_realization_seed_matches_frozen_audits(self) -> None:
        self.assertEqual(realization_seed("exact", 14, 2), 200014)
        self.assertEqual(realization_seed("learned", 14, 2), 700014)
        with self.assertRaisesRegex(ValueError, "mode"):
            realization_seed("other", 14, 2)

    def test_ffmpeg_contract_is_browser_compatible(self) -> None:
        command = ffmpeg_command(
            "/usr/bin/ffmpeg", Path("video.mp4"), width=216, height=144, fps=12
        )
        self.assertIn("libx264", command)
        self.assertIn("yuv420p", command)
        self.assertIn("+faststart", command)
        self.assertEqual(command[-1], "video.mp4")
        with self.assertRaisesRegex(ValueError, "positive"):
            ffmpeg_command("ffmpeg", Path("x.mp4"), width=0, height=1, fps=1)


if __name__ == "__main__":
    unittest.main()
