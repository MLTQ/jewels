"""Tests for causal background and seam metrics in the three-window renderer."""

from __future__ import annotations

import unittest

import torch

from sol.render_scaffold_mark_rollout import (
    BASE_PANELS,
    _appearance_saliency_gates,
    _base_lock_report,
    _causal_background,
    _panel_names,
    _seam_report,
    _source_seed_map,
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

    def test_source_seeds_do_not_change_when_a_filtered_subset_is_rendered(self) -> None:
        seeds = _source_seed_map(("basketball", "horse", "guitar", "eye"), 31)
        self.assertEqual(seeds["basketball"], 31)
        self.assertEqual(seeds["horse"], 32)
        self.assertEqual(seeds["guitar"], 33)
        self.assertEqual(seeds["eye"], 34)

    def test_base_lock_allows_only_named_augmented_state(self) -> None:
        base = {"trunk.weight": torch.ones(2), "head.bias": torch.zeros(1)}
        candidate = {
            **{name: value.clone() for name, value in base.items()},
            "set_blocks.0.weight": torch.randn(2),
        }
        report = _base_lock_report(
            base, candidate, added_prefixes=("set_blocks.",)
        )
        self.assertTrue(report["shared_tensors_exact"])
        candidate["trunk.weight"][0] = 2
        with self.assertRaisesRegex(ValueError, "modified"):
            _base_lock_report(base, candidate, added_prefixes=("set_blocks.",))


if __name__ == "__main__":
    unittest.main()
