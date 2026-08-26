"""Tests for scene/block hierarchy causal control ownership."""

from __future__ import annotations

import unittest

import torch

from sol.audit_scene_block_constellation_oracle import control_conditions


class SceneBlockConstellationOracleAuditTests(unittest.TestCase):
    def test_controls_disrupt_global_and_local_levels_separately(self) -> None:
        program = torch.tensor([1, 2, 3])
        shuffled = torch.tensor([4, 5, 6])
        arms = control_conditions(
            1, program, shuffled,
            semantic_scenes=3, null_scene=3, null_token=9,
        )
        self.assertEqual(arms["oracle hierarchy"][0], 1)
        self.assertEqual(arms["shuffled scene"][0], 2)
        self.assertTrue(torch.equal(arms["shuffled scene"][1], program))
        self.assertEqual(arms["shuffled blocks"][0], 1)
        self.assertTrue(torch.equal(arms["shuffled blocks"][1], shuffled))
        self.assertEqual(arms["null hierarchy"][0], 3)
        self.assertEqual(arms["null hierarchy"][1].tolist(), [9, 9, 9])


if __name__ == "__main__":
    unittest.main()
