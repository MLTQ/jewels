"""Tests for conservative block masks and hierarchical clamping."""

from __future__ import annotations

import unittest

import torch

from sol.hierarchical_inpaint import (
    coarsen_dirty_mask,
    expand_coarse_mask,
    hierarchical_masked_flow_inpaint,
    restore_clean_codes,
)


def _velocity(
    state: torch.Tensor,
    _time: torch.Tensor,
    _condition: torch.Tensor | None,
) -> torch.Tensor:
    return torch.ones_like(state)


class HierarchicalInpaintTests(unittest.TestCase):
    def test_mask_roundtrip_expands_only_touched_blocks(self) -> None:
        shape = (4, 4, 4)
        fine = torch.zeros(64, dtype=torch.bool)
        fine[0] = True
        fine[63] = True
        coarse = coarsen_dirty_mask(fine, shape, 2)
        self.assertEqual(int(coarse.sum()), 2)
        expanded = expand_coarse_mask(coarse, shape, 2)
        self.assertEqual(int(expanded.sum()), 16)
        self.assertTrue(bool(expanded[fine].all()))

    def test_batched_mask_layout_matches_single_masks(self) -> None:
        first = torch.zeros(64, dtype=torch.bool)
        second = torch.zeros(64, dtype=torch.bool)
        first[5] = True
        second[37] = True
        batched = coarsen_dirty_mask(torch.stack([first, second]), (4, 4, 4), 2)
        torch.testing.assert_close(
            batched[0], coarsen_dirty_mask(first, (4, 4, 4), 2)
        )
        torch.testing.assert_close(
            batched[1], coarsen_dirty_mask(second, (4, 4, 4), 2)
        )

    def test_hierarchical_sampling_preserves_clean_codes_exactly(self) -> None:
        known = torch.randn(1, 8, 5)
        fine = torch.zeros(64, dtype=torch.bool)
        fine[0] = True
        result = hierarchical_masked_flow_inpaint(
            _velocity,
            known,
            fine,
            (4, 4, 4),
            2,
            steps=3,
            generator=torch.Generator().manual_seed(4),
        )
        clean = ~result.dirty_coarse
        torch.testing.assert_close(
            result.normalized_coarse[:, clean], known[:, clean], rtol=0, atol=0
        )
        self.assertGreater(
            float((result.normalized_coarse[:, ~clean] - known[:, ~clean]).abs().max()),
            0,
        )

    def test_invalid_block_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            coarsen_dirty_mask(torch.zeros(60), (3, 4, 5), 2)

    def test_raw_clean_codes_are_restored_bit_exactly(self) -> None:
        known = torch.randn(2, 8, 3)
        repaired = known + 0.001
        dirty = torch.zeros(8, dtype=torch.bool)
        dirty[[1, 6]] = True
        restored = restore_clean_codes(repaired, known, dirty)
        torch.testing.assert_close(restored[:, ~dirty], known[:, ~dirty], rtol=0, atol=0)
        torch.testing.assert_close(restored[:, dirty], repaired[:, dirty], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
