"""Tests for trajectory-speaker evidence plotting utilities."""

from __future__ import annotations

import unittest

from PIL import Image

from sol.plot_trajectory_speaker_evidence import _middle_panel


class TrajectorySpeakerEvidencePlotTests(unittest.TestCase):
    def test_middle_panel_crop(self) -> None:
        image = Image.new("RGB", (654, 1536), "white")
        panel = _middle_panel(image, 4)
        self.assertEqual(panel.size, (218, 168))


if __name__ == "__main__":
    unittest.main()
