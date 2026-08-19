"""Tests for structural distillation helpers."""

from __future__ import annotations

import unittest

import torch

from sol.distill_structural_encoder import chamfer, teacher_descriptors


class DistillHelperTests(unittest.TestCase):
    def test_chamfer_is_zero_for_identical_sets(self) -> None:
        points = torch.rand(64, 3)
        value, index = chamfer(points, points.clone(), chunk=16)
        self.assertLess(float(value), 1e-6)  # float32 cdist noise, not exact zero
        self.assertTrue(torch.equal(index, torch.arange(64)))

    def test_chamfer_grows_with_separation(self) -> None:
        a = torch.rand(32, 3)
        near, _ = chamfer(a, a + 0.01, chunk=16)
        far, _ = chamfer(a, a + 0.5, chunk=16)
        self.assertGreater(float(far), float(near))

    def test_chamfer_penalises_uncovered_teacher_regions(self) -> None:
        """Teacher->student direction must punish ignoring a dense cluster."""
        teacher = torch.cat((torch.rand(32, 3) * 0.1, torch.rand(32, 3) * 0.1 + 0.9))
        covering = teacher.clone()
        one_sided = torch.rand(64, 3) * 0.1
        full, _ = chamfer(covering, teacher, chunk=16)
        partial, _ = chamfer(one_sided, teacher, chunk=16)
        self.assertGreater(float(partial), float(full))

    def test_teacher_descriptors_return_centres_and_spreads(self) -> None:
        torch.manual_seed(0)
        features = torch.randn(500, 22)
        features[:, 3:9] *= 0.3
        centres, spread = teacher_descriptors(
            features, 100, torch.Generator().manual_seed(0)
        )
        self.assertEqual(centres.shape, (100, 3))
        self.assertEqual(spread.shape, (100,))
        self.assertTrue(bool((spread >= 0).all()))


if __name__ == "__main__":
    unittest.main()
