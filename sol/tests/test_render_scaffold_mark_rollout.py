"""Tests for causal background and seam metrics in the three-window renderer."""

from __future__ import annotations

import unittest

import torch

from sol.render_scaffold_mark_rollout import _causal_background, _seam_report


class RenderScaffoldMarkRolloutTests(unittest.TestCase):
    def test_background_uses_only_initial_stride(self) -> None:
        initial = torch.full((4, 2, 3, 3), 0.25)
        self.assertTrue(torch.equal(_causal_background(initial), torch.full((3,), 0.25)))

    def test_seam_report_finds_stride_boundaries(self) -> None:
        target = torch.zeros(12, 2, 2, 3)
        candidate = target.clone()
        candidate[4:] += 0.2
        candidate[8:] += 0.2
        report = _seam_report(candidate, target, 4)
        self.assertAlmostEqual(report["candidate_seam_change"], 0.2, places=6)
        self.assertGreater(report["seam_to_regular_ratio"], 1e6)


if __name__ == "__main__":
    unittest.main()
