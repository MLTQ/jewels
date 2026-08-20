"""Regression tests for the production spacetime primitive renderer."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


STPRIM_ROOT = Path(__file__).resolve().parents[2] / "stprim"
if str(STPRIM_ROOT) not in sys.path:
    sys.path.insert(0, str(STPRIM_ROOT))

from core.params import PrimitiveField  # noqa: E402
from fit.fitter import project_geometry_  # noqa: E402
from models.render import SupportOverflowError, render_points  # noqa: E402


def elongated_counterexample() -> tuple[PrimitiveField, torch.Tensor]:
    """Build 65 near zero splats plus one far, time-tilted contributor."""
    field = PrimitiveField(66, p1_color=False)
    angles = torch.linspace(0.0, 2.0 * math.pi, 66)[:-1]
    with torch.no_grad():
        field.mu[:65, 0] = 0.1 * angles.cos()
        field.mu[:65, 1] = 0.1 * angles.sin()
        field.mu[:65, 2] = 0.0
        field.mu[65] = torch.tensor([0.8, 0.0, 0.8])
        field.log_scale[:65] = math.log(0.01)
        field.log_scale[65] = torch.tensor(
            [math.log(2.0), math.log(0.01), math.log(0.01)]
        )
        field.quat.zero_()
        field.quat[:, 0] = 1.0
        half_angle = -math.pi / 8.0
        field.quat[65, 0] = math.cos(half_angle)
        field.quat[65, 2] = math.sin(half_angle)
        field.color.zero_()
        field.color[65] = 1.0
        field.logit_w.fill_(10.0)
    return field, torch.zeros(1, 3)


class ProductionRenderTests(unittest.TestCase):
    def test_support_mode_keeps_contributor_missed_by_center_knn(self) -> None:
        field, point = elongated_counterexample()

        exact = render_points(field, point, cull_mode="exact")
        legacy = render_points(field, point, cull_mode="knn", knn=64)
        support = render_points(
            field,
            point,
            cull_mode="support",
            support_sigma=5.0,
            support_capacity=16,
        )

        self.assertGreater(float((exact - legacy).abs().max().detach()), 0.8)
        torch.testing.assert_close(support, exact, atol=1e-6, rtol=1e-6)

    def test_support_mode_fails_instead_of_truncating_candidates(self) -> None:
        field = PrimitiveField(8, p1_color=False, init_scale=1.0)
        with torch.no_grad():
            field.mu.zero_()

        with self.assertRaisesRegex(SupportOverflowError, "capacity 4"):
            render_points(
                field,
                torch.zeros(2, 3),
                cull_mode="support",
                support_capacity=4,
            )

    def test_support_mode_preserves_parameter_gradients(self) -> None:
        field, point = elongated_counterexample()
        render_points(
            field,
            point,
            cull_mode="support",
            support_capacity=16,
        ).sum().backward()

        self.assertGreater(float(field.color.grad[65].abs().sum()), 0.0)
        self.assertGreater(float(field.mu.grad[65].abs().sum()), 0.0)

    def test_geometry_controls_project_rotation_and_scale(self) -> None:
        field = PrimitiveField(3, p1_color=False)
        with torch.no_grad():
            field.log_scale[:] = torch.log(torch.tensor([0.1, 0.2, 0.4]))

        project_geometry_(field, "axis_aligned")
        expected_rotation = torch.eye(3).expand(3, -1, -1)
        torch.testing.assert_close(field.rotations(), expected_rotation)
        self.assertFalse(bool((field.scales()[:, 0] == field.scales()[:, 1]).all()))

        project_geometry_(field, "isotropic")
        torch.testing.assert_close(field.rotations(), expected_rotation)
        scale = field.scales()
        torch.testing.assert_close(scale[:, :1].expand_as(scale), scale)

        with self.assertRaisesRegex(ValueError, "unknown geometry_constraint"):
            project_geometry_(field, "invalid")


if __name__ == "__main__":
    unittest.main()
