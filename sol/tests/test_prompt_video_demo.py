"""Tests for the prompt demo's browser-facing helpers."""

from __future__ import annotations

import unittest

from sol.prompt_video_demo import demo_html, parse_byte_range


class PromptVideoDemoTests(unittest.TestCase):
    def test_browser_byte_ranges(self) -> None:
        self.assertIsNone(parse_byte_range(None, 100))
        self.assertEqual(parse_byte_range("bytes=10-19", 100), (10, 19))
        self.assertEqual(parse_byte_range("bytes=90-", 100), (90, 99))
        self.assertEqual(parse_byte_range("bytes=-5", 100), (95, 99))
        with self.assertRaises(ValueError):
            parse_byte_range("bytes=100-", 100)
        with self.assertRaises(ValueError):
            parse_byte_range("bytes=0-1,4-5", 100)

    def test_page_exposes_examples_and_truthful_modes(self) -> None:
        prompt = "a ballerina spinning"
        page = demo_html((prompt, "a dog catching a ball"), True)
        self.assertIn(prompt, page)
        self.assertIn("Proven prompts", page)
        self.assertIn("Free-form wording", page)
        self.assertIn("not yet", page)
        disabled = demo_html((prompt,), False)
        self.assertIn("optional learned speaker checkpoint is not installed", disabled)
        self.assertIn("disabled", disabled)


if __name__ == "__main__":
    unittest.main()
