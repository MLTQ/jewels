"""Tests for frozen fine block-language settings."""

from __future__ import annotations

import unittest

from sol.fit_fine_block_language import validate_fine_language_settings


class FineBlockLanguageFitTests(unittest.TestCase):
    def test_registered_settings_are_accepted(self) -> None:
        validate_fine_language_settings((16, 16, 8), 1024)

    def test_unregistered_shape_or_capacity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen"):
            validate_fine_language_settings((8, 8, 4), 1024)
        with self.assertRaisesRegex(ValueError, "frozen"):
            validate_fine_language_settings((16, 16, 8), 2048)


if __name__ == "__main__":
    unittest.main()
