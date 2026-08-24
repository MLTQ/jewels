"""Tests for checkpoint-level frozen-geometry comparisons."""

from __future__ import annotations

import unittest

import torch

from sol.compare_frozen_geometry import compare_geometry_states, geometry_state


class FrozenGeometryComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "geometry_trunk.0.weight": torch.zeros(2, 2),
            "geometry_head.bias": torch.ones(2),
            "appearance_head.bias": torch.full((2,), 7.0),
        }

    def test_appearance_changes_do_not_affect_geometry_result(self) -> None:
        candidate = {name: value.clone() for name, value in self.state.items()}
        candidate["appearance_head.bias"].add_(1)
        report = compare_geometry_states(self.state, candidate)
        self.assertTrue(report["bitwise_exact"])
        self.assertEqual(report["tensor_count"], 2)

    def test_one_ulp_geometry_change_is_reported(self) -> None:
        candidate = {name: value.clone() for name, value in self.state.items()}
        candidate["geometry_head.bias"][0] = torch.nextafter(
            candidate["geometry_head.bias"][0], torch.tensor(2.0)
        )
        report = compare_geometry_states(self.state, candidate)
        self.assertFalse(report["bitwise_exact"])
        self.assertEqual(report["mismatched_tensors"], ["geometry_head.bias"])

    def test_missing_geometry_contract_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no factorized geometry"):
            geometry_state({"appearance_head.bias": torch.zeros(1)})


if __name__ == "__main__":
    unittest.main()
