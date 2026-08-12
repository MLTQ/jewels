"""Tests for causal background and seam metrics in the three-window renderer."""

from __future__ import annotations

import unittest

import torch

from sol.render_scaffold_mark_rollout import (
    BASE_PANELS,
    _appearance_saliency_gates,
    _causal_background,
    _panel_names,
    _seam_report,
)
from sol.lifecycle_appearance_flow import APPEARANCE_DIMENSION_SETS


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

    def test_paired_panel_names_preserve_baseline_and_add_frozen_control(self) -> None:
        self.assertEqual(_panel_names(False), BASE_PANELS)
        paired = _panel_names(True)
        self.assertEqual(len(paired), len(BASE_PANELS) + 1)
        self.assertEqual(paired[2], "generated frozen base")
        self.assertNotIn(21, APPEARANCE_DIMENSION_SETS["static-detail"])
        self.assertNotIn(20, APPEARANCE_DIMENSION_SETS["static-detail"])

    def test_appearance_saliency_gate_selects_declared_fraction(self) -> None:
        guide = torch.zeros(8, 3)
        guide[3] = 1
        gates = _appearance_saliency_gates(
            (guide,), (2, 2, 2), torch.zeros(3), 0.25
        )
        self.assertEqual(int(gates[0].sum()), 2)
        self.assertEqual(float(gates[0][3]), 1.0)


if __name__ == "__main__":
    unittest.main()
