"""Tests for learned-topology to frozen-realizer contract bridging."""

from __future__ import annotations

import unittest

import torch

from sol.scaffold_topology_realizer import validate_realizer_topology
from sol.token_grid import GridCapacityError, GridSpec


class ScaffoldTopologyRealizerTests(unittest.TestCase):
    def test_valid_counts_expand_for_the_realizer(self) -> None:
        topology_spec = GridSpec((2, 2, 2), 16)
        realizer_spec = GridSpec((2, 2, 2), 4)
        counts = torch.tensor([2, 0, 3, 1, 0, 0, 4, 1])
        decoded = validate_realizer_topology(
            counts, topology_spec, realizer_spec
        )
        self.assertTrue(torch.equal(decoded.counts, counts))
        self.assertEqual(len(decoded.cell_indices), int(counts.sum()))
        self.assertTrue(
            torch.equal(decoded.slot_indices[-5:], torch.tensor([0, 1, 2, 3, 0]))
        )

    def test_grid_mismatch_and_capacity_overflow_are_hard_errors(self) -> None:
        topology_spec = GridSpec((2, 2, 2), 16)
        with self.assertRaisesRegex(ValueError, "grid shapes"):
            validate_realizer_topology(
                torch.ones(8, dtype=torch.long),
                topology_spec,
                GridSpec((2, 2, 1), 8),
            )
        with self.assertRaisesRegex(GridCapacityError, "needs 5 ranks"):
            validate_realizer_topology(
                torch.tensor([5, 0, 0, 0, 0, 0, 0, 0]),
                topology_spec,
                GridSpec((2, 2, 2), 4),
            )


if __name__ == "__main__":
    unittest.main()
