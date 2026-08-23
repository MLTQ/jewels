"""Tests for local fitted-teacher attribute correspondence and losses."""

from __future__ import annotations

import math
import unittest

import torch

from sol.local_teacher_distillation import (
    LocalTeacherAttributes,
    extract_local_teacher_attributes,
    local_teacher_attribute_losses,
    renderer_responsibility_targets,
    responsibility_teacher_moment_losses,
    soft_local_correspondence,
)


class LocalTeacherDistillationTests(unittest.TestCase):
    def test_extract_preserves_canonical_attributes(self) -> None:
        features = torch.zeros(6, 22)
        features[:, :3] = torch.arange(18).reshape(6, 3) / 20
        features[:, 9:12] = 0.25
        features[:, 12:21] = 0.1
        features[:4, 21] = 0.0
        features[4:, 21] = -10.0
        teacher = extract_local_teacher_attributes(
            features, 6, torch.Generator().manual_seed(0)
        )
        self.assertEqual(teacher.centers.shape, (6, 3))
        self.assertEqual(teacher.covariance.shape, (6, 3, 3))
        self.assertEqual(teacher.precision.shape, (6, 3, 3))
        self.assertEqual(teacher.log_scale.shape, (6, 3))
        self.assertEqual(teacher.axis.shape, (6, 3))
        self.assertEqual(teacher.color_grads.shape, (6, 3, 3))
        self.assertEqual(teacher.active_count, 4.0)
        self.assertTrue(torch.allclose(teacher.colors, torch.full((6, 3), 0.25)))

    def test_correspondence_prefers_nearby_teacher_and_is_detached(self) -> None:
        student = torch.tensor([[0.01, 0.0, 0.0]], requires_grad=True)
        teacher = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        index, weight = soft_local_correspondence(
            student, teacher, neighbors=2, temperature=0.1
        )
        self.assertEqual(index.tolist(), [[0, 1]])
        self.assertGreater(float(weight[0, 0]), 0.99)
        self.assertFalse(weight.requires_grad)

    def test_matching_attributes_have_near_zero_loss_and_no_center_gradient(self) -> None:
        opacity = 0.4
        teacher = LocalTeacherAttributes(
            centers=torch.zeros(1, 3),
            covariance=torch.eye(3)[None],
            precision=torch.eye(3)[None],
            log_scale=torch.tensor([[-2.0, -1.5, -1.0]]),
            axis=torch.tensor([[1.0, 0.0, 0.0]]),
            opacity=torch.tensor([opacity]),
            colors=torch.tensor([[0.2, 0.4, 0.6]]),
            color_grads=torch.full((1, 3, 3), 0.1),
            active_count=1.0,
        )
        centers = torch.zeros(1, 3, requires_grad=True)
        colors = teacher.colors.clone().requires_grad_()
        losses = local_teacher_attribute_losses(
            student_centers=centers,
            student_log_scale=teacher.log_scale[:, [2, 0, 1]],
            student_axis=-teacher.axis,
            student_opacity=torch.tensor([opacity]),
            student_colors=colors,
            student_color_grads=teacher.color_grads.clone(),
            teacher=teacher,
            neighbors=1,
            temperature=0.1,
            size_offset=0.0,
            opacity_mass_ratio=1.0,
        )
        total = sum(losses.values())
        self.assertLess(float(total.detach()), 1e-8)
        total.backward()
        self.assertIsNone(centers.grad)
        self.assertIsNotNone(colors.grad)

    def test_opacity_target_matches_compensated_optical_mass(self) -> None:
        teacher_opacity = 0.2
        ratio = 3.0
        target = 1.0 - (1.0 - teacher_opacity) ** ratio
        teacher = LocalTeacherAttributes(
            centers=torch.zeros(1, 3),
            covariance=torch.eye(3)[None],
            precision=torch.eye(3)[None],
            log_scale=torch.zeros(1, 3),
            axis=torch.tensor([[0.0, 0.0, 1.0]]),
            opacity=torch.tensor([teacher_opacity]),
            colors=torch.zeros(1, 3),
            color_grads=torch.zeros(1, 3, 3),
            active_count=3.0,
        )
        losses = local_teacher_attribute_losses(
            student_centers=torch.zeros(1, 3),
            student_log_scale=torch.zeros(1, 3),
            student_axis=teacher.axis,
            student_opacity=torch.tensor([target]),
            student_colors=teacher.colors,
            student_color_grads=teacher.color_grads,
            teacher=teacher,
            neighbors=1,
            temperature=0.1,
            size_offset=0.0,
            opacity_mass_ratio=ratio,
        )
        self.assertTrue(math.isclose(float(losses["opacity"]), 0.0, abs_tol=1e-7))

    def test_active_uniform_sampling_excludes_inactive_jewels(self) -> None:
        features = torch.zeros(8, 22)
        features[:4, 21] = 0.0
        features[4:, 21] = -10.0
        features[:, 9] = torch.arange(8)
        teacher = extract_local_teacher_attributes(
            features, 3, torch.Generator().manual_seed(2),
            sampling="active_uniform",
        )
        self.assertTrue(bool((teacher.colors[:, 0] < 4).all()))

    def test_responsibility_uses_covariance_support_not_center_distance(self) -> None:
        covariance = torch.stack((
            torch.diag(torch.tensor([1.0, 0.01, 0.01])),
            torch.eye(3) * 0.001,
        ))
        teacher = LocalTeacherAttributes(
            centers=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.1, 0.0]]),
            covariance=covariance,
            precision=torch.linalg.inv(covariance),
            log_scale=0.5 * torch.linalg.eigvalsh(covariance).log(),
            axis=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            opacity=torch.full((2,), 0.5),
            colors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            color_grads=torch.zeros(2, 3, 3),
            active_count=2.0,
        )
        targets = renderer_responsibility_targets(
            torch.tensor([[0.8, 0.0, 0.0]], requires_grad=True), teacher,
            support_sigma=5.0, temperature=1.0,
        )
        self.assertGreater(float(targets.colors[0, 0]), 0.99)
        self.assertLess(float(targets.colors[0, 1]), 0.01)
        self.assertFalse(targets.colors.requires_grad)
        self.assertFalse(bool(targets.used_fallback[0]))

    def test_responsibility_color_and_jacobian_match_single_teacher(self) -> None:
        gradient = torch.diag(torch.tensor([0.1, 0.2, 0.3]))[None]
        teacher = LocalTeacherAttributes(
            centers=torch.zeros(1, 3),
            covariance=torch.eye(3)[None],
            precision=torch.eye(3)[None],
            log_scale=torch.zeros(1, 3),
            axis=torch.tensor([[0.0, 0.0, 1.0]]),
            opacity=torch.tensor([0.5]),
            colors=torch.tensor([[0.2, 0.3, 0.4]]),
            color_grads=gradient,
            active_count=1.0,
        )
        targets = renderer_responsibility_targets(
            torch.tensor([[0.5, 0.0, 0.0]]), teacher,
            support_sigma=5.0, temperature=1.0,
        )
        torch.testing.assert_close(targets.colors, torch.tensor([[0.25, 0.3, 0.4]]))
        torch.testing.assert_close(targets.color_grads, gradient)

    def test_matching_responsibility_moments_have_near_zero_losses(self) -> None:
        covariance = torch.eye(3)[None] * 0.25
        teacher = LocalTeacherAttributes(
            centers=torch.zeros(1, 3),
            covariance=covariance,
            precision=torch.linalg.inv(covariance),
            log_scale=torch.full((1, 3), math.log(0.5)),
            axis=torch.tensor([[0.0, 0.0, 1.0]]),
            opacity=torch.tensor([0.4]),
            colors=torch.tensor([[0.2, 0.3, 0.4]]),
            color_grads=torch.full((1, 3, 3), 0.1),
            active_count=1.0,
        )
        losses, targets = responsibility_teacher_moment_losses(
            student_centers=torch.zeros(1, 3, requires_grad=True),
            student_log_scale=teacher.log_scale,
            student_axis=-teacher.axis,
            student_opacity=teacher.opacity,
            student_colors=teacher.colors,
            student_color_grads=teacher.color_grads,
            teacher=teacher,
            support_sigma=5.0,
            temperature=1.0,
            size_offset=0.0,
            opacity_mass_ratio=1.0,
        )
        self.assertLess(float(sum(losses.values())), 1e-8)
        self.assertEqual(float(targets.effective_count), 1.0)
        self.assertEqual(float(targets.support_count), 1.0)

    def test_responsibility_marks_nearest_mahalanobis_fallback(self) -> None:
        teacher = LocalTeacherAttributes(
            centers=torch.zeros(1, 3),
            covariance=torch.eye(3)[None] * 0.01,
            precision=torch.eye(3)[None] * 100,
            log_scale=torch.full((1, 3), math.log(0.1)),
            axis=torch.tensor([[0.0, 0.0, 1.0]]),
            opacity=torch.tensor([0.5]),
            colors=torch.zeros(1, 3),
            color_grads=torch.zeros(1, 3, 3),
            active_count=1.0,
        )
        targets = renderer_responsibility_targets(
            torch.ones(1, 3), teacher, support_sigma=1.0, temperature=1.0
        )
        self.assertTrue(bool(targets.used_fallback[0]))
        self.assertEqual(float(targets.support_count), 1.0)


if __name__ == "__main__":
    unittest.main()
