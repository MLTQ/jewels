"""Tests for local fitted-teacher attribute correspondence and losses."""

from __future__ import annotations

import math
import unittest

import torch

from sol.local_teacher_distillation import (
    LocalTeacherAttributes,
    extract_local_teacher_attributes,
    local_teacher_attribute_losses,
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


if __name__ == "__main__":
    unittest.main()
