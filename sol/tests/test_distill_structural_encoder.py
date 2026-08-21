"""Tests for structural distillation helpers."""

from __future__ import annotations

import unittest

import torch

from sol.distill_structural_encoder import (
    chamfer,
    freeze_geometry_state,
    mask_geometry_gradients,
    mixed_spacetime_tilt,
    orientation_loss,
    principal_axis,
    schedule_multiplier,
    select_validation_ids,
    soft_active_fraction,
    soft_occupancy,
    teacher_descriptors,
    restore_geometry_state,
)
from sol.structural_encoder import StructuralJewelEncoder
from sol.token_grid import GridSpec


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
        centres, spread, axis, weight = teacher_descriptors(
            features, 100, torch.Generator().manual_seed(0)
        )
        self.assertEqual(centres.shape, (100, 3))
        self.assertEqual(spread.shape, (100,))
        self.assertEqual(axis.shape, (100, 3))
        self.assertEqual(weight.shape, (100,))
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

    def test_mixed_tilt_distinguishes_diagonal_from_pure_axes(self) -> None:
        axes = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [2 ** -0.5, 0.0, 2 ** -0.5],
        ])
        tilt = mixed_spacetime_tilt(axes)
        torch.testing.assert_close(tilt, torch.tensor([0.0, 0.0, 1.0]), atol=1e-4, rtol=0)


class DensityMatchingTests(unittest.TestCase):
    def test_soft_occupancy_sums_to_one(self) -> None:
        from sol.token_grid import GridSpec

        grid = GridSpec((4, 4, 2), 1)
        occupancy = soft_occupancy(torch.rand(200, 3) * 2 - 1, grid)
        self.assertEqual(occupancy.shape, (grid.n_cells,))
        self.assertAlmostEqual(float(occupancy.sum()), 1.0, places=4)

    def test_opacity_weights_change_occupancy(self) -> None:
        from sol.token_grid import GridSpec

        grid = GridSpec((2, 1, 1), 1)
        points = torch.tensor([[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]])
        uniform = soft_occupancy(points, grid)
        weighted = soft_occupancy(points, grid, weights=torch.tensor([9.0, 1.0]))
        self.assertAlmostEqual(float(uniform[0]), float(uniform[1]), places=4)
        self.assertGreater(float(weighted[0]), float(weighted[1]) * 5)

    def test_soft_active_fraction_tracks_opacity_floor(self) -> None:
        logits = torch.logit(torch.tensor([0.001, 0.001, 0.5, 0.5]))
        fraction = soft_active_fraction(logits, temperature=0.1)
        self.assertAlmostEqual(float(fraction), 0.5, places=3)

    def test_density_loss_is_small_for_matching_distributions(self) -> None:
        from sol.distill_structural_encoder import density_loss
        from sol.token_grid import GridSpec

        grid = GridSpec((4, 4, 2), 1)
        torch.manual_seed(0)
        points = torch.rand(400, 3) * 2 - 1
        same = float(density_loss(points, points.clone(), grid))
        clustered = torch.rand(400, 3) * 0.3 - 1.0
        different = float(density_loss(clustered, points, grid))
        self.assertLess(same, 1e-4)
        self.assertGreater(different, same + 0.5)

    def test_density_loss_gradient_pushes_toward_teacher(self) -> None:
        from sol.distill_structural_encoder import density_loss
        from sol.token_grid import GridSpec

        grid = GridSpec((4, 4, 2), 1)
        torch.manual_seed(1)
        teacher = torch.rand(500, 3) * 0.4 + 0.5      # teacher clustered high
        student = (torch.rand(500, 3) * 2 - 1).requires_grad_(True)
        density_loss(student, teacher, grid).backward()
        self.assertIsNotNone(student.grad)
        self.assertGreater(float(student.grad.abs().sum()), 0.0)


class ValidationSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        video = torch.empty(0)
        self.videos = {
            "train": (video, "train"),
            "v1": (video, "validation"),
            "v2": (video, "validation"),
        }

    def test_explicit_validation_order_and_limit(self) -> None:
        self.assertEqual(
            select_validation_ids(self.videos, ("v2", "v1"), 1), ["v2"]
        )

    def test_missing_validation_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unavailable"):
            select_validation_ids(self.videos, ("missing",), 0)

    def test_delayed_schedule_is_zero_then_ramps_to_one(self) -> None:
        self.assertEqual(schedule_multiplier(100, 100, 200), 0.0)
        self.assertEqual(schedule_multiplier(200, 100, 200), 0.5)
        self.assertEqual(schedule_multiplier(300, 100, 200), 1.0)
        self.assertEqual(schedule_multiplier(500, 100, 200), 1.0)


class FrozenGeometryTests(unittest.TestCase):
    def test_freeze_masks_and_restores_only_geometry_rows(self) -> None:
        model = StructuralJewelEncoder(
            grid_spec=GridSpec((2, 2, 2), 1), slots_per_cell=2, model_dim=8
        )
        frozen = freeze_geometry_state(model)
        self.assertTrue(all(not p.requires_grad for p in model.trunk.parameters()))
        model.head.weight.grad = torch.ones_like(model.head.weight)
        model.head.bias.grad = torch.ones_like(model.head.bias)
        mask_geometry_gradients(model, frozen)
        rows = frozen["rows"]
        self.assertEqual(float(model.head.weight.grad[rows].abs().sum()), 0.0)
        self.assertGreater(float(model.head.weight.grad[~rows].abs().sum()), 0.0)

        before_appearance = model.head.weight.detach()[~rows].clone()
        with torch.no_grad():
            model.head.weight.add_(1.0)
            model.head.bias.add_(1.0)
        restore_geometry_state(model, frozen)
        torch.testing.assert_close(model.head.weight[rows], frozen["weight"])
        torch.testing.assert_close(model.head.bias[rows], frozen["bias"])
        torch.testing.assert_close(
            model.head.weight[~rows], before_appearance + 1.0
        )


if __name__ == "__main__":
    unittest.main()
