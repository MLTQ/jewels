"""Tests for structural distillation helpers."""

from __future__ import annotations

import unittest

import torch

from sol.distill_structural_encoder import (
    chamfer,
    orientation_loss,
    principal_axis,
    teacher_descriptors,
)


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

    def test_teacher_descriptors_return_centres_spreads_and_axes(self) -> None:
        torch.manual_seed(0)
        features = torch.randn(500, 22)
        features[:, 3:9] *= 0.3
        centres, spread, axis = teacher_descriptors(
            features, 100, torch.Generator().manual_seed(0)
        )
        self.assertEqual(centres.shape, (100, 3))
        self.assertEqual(spread.shape, (100,))
        self.assertEqual(axis.shape, (100, 3))
        self.assertTrue(bool((spread >= 0).all()))
        self.assertTrue(torch.allclose(axis.norm(dim=-1), torch.ones(100), atol=1e-5))


class OrientationTests(unittest.TestCase):
    def test_principal_axis_follows_the_largest_scale(self) -> None:
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        for largest, expected in ((0, [1, 0, 0]), (1, [0, 1, 0]), (2, [0, 0, 1])):
            log_scale = torch.full((1, 3), -2.0)
            log_scale[0, largest] = 1.0
            axis = principal_axis(quaternion, log_scale)
            self.assertTrue(
                torch.allclose(axis.abs(), torch.tensor([expected], dtype=torch.float), atol=1e-5)
            )

    def test_orientation_loss_is_zero_when_aligned_and_sign_invariant(self) -> None:
        axis = torch.nn.functional.normalize(torch.randn(20, 3), dim=-1)
        self.assertLess(float(orientation_loss(axis, axis)), 1e-6)
        self.assertLess(float(orientation_loss(axis, -axis)), 1e-6)

    def test_orientation_loss_penalises_perpendicular_axes(self) -> None:
        a = torch.tensor([[1.0, 0.0, 0.0]])
        b = torch.tensor([[0.0, 1.0, 0.0]])
        self.assertGreater(float(orientation_loss(a, b)), 0.9)


if __name__ == "__main__":
    unittest.main()
