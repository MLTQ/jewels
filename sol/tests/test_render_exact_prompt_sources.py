"""Tests for exact-prompt source-video matching."""

from __future__ import annotations

import unittest

from sol.render_exact_prompt_sources import match_sources


class ExactPromptSourceRenderTests(unittest.TestCase):
    def test_matches_exact_style_and_prompt_only(self) -> None:
        target = {
            "examples": [
                {"split": "train", "style": "anime", "source_prompt": "dance", "source_id": "target"},
                {"split": "train", "style": "cartoon", "source_prompt": "dance", "source_id": "wrong-style"},
            ]
        }
        exact = {
            "examples": [
                {"style": "anime", "source_prompt": "dance", "source_id": "new-b"},
                {"style": "anime", "source_prompt": "dance", "source_id": "new-a"},
            ]
        }
        matches = match_sources(target, exact)
        self.assertEqual(matches[0][0]["source_id"], "target")
        self.assertEqual([row["source_id"] for row in matches[0][1]], ["new-a", "new-b"])

    def test_accepts_more_than_two_new_sources(self) -> None:
        target = {
            "examples": [
                {"split": "train", "style": "anime", "source_prompt": "dance", "source_id": "target"},
            ]
        }
        exact = {
            "examples": [
                {"style": "anime", "source_prompt": "dance", "source_id": f"new-{index}"}
                for index in range(4)
            ]
        }
        self.assertEqual(len(match_sources(target, exact)[0][1]), 4)

    def test_requires_at_least_two_new_sources(self) -> None:
        target = {
            "examples": [
                {"split": "train", "style": "anime", "source_prompt": "dance", "source_id": "target"},
            ]
        }
        exact = {
            "examples": [
                {"style": "anime", "source_prompt": "dance", "source_id": "new"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "at least two new"):
            match_sources(target, exact)


if __name__ == "__main__":
    unittest.main()
