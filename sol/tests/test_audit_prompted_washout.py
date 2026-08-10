"""Prompted washout decomposition tests."""

from __future__ import annotations

import math
import unittest

import torch

from sol.audit_prompted_washout import (
    render_signature,
    replace_groups,
    topology_adherence,
)
from sol.token_grid import GridSpec


def _features() -> torch.Tensor:
    values = torch.zeros(2, 22)
    values[:, :3] = torch.tensor([[-0.75, -0.75, 0.25], [0.75, 0.75, 0.25]])
    values[:, 3] = 2 * math.log(0.05)
    values[:, 6] = 2 * math.log(0.05)
    values[:, 8] = 2 * math.log(0.04)
    values[:, 9:12] = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    values[:, 21] = 2.0
    return values


class PromptedWashoutAuditTests(unittest.TestCase):
    def test_replace_groups_only_changes_requested_features(self) -> None:
        target = torch.zeros(3, 22)
        source = torch.ones(3, 22)
        hybrid = replace_groups(target, source, ("center", "opacity"))
        self.assertTrue(torch.equal(hybrid[:, :3], source[:, :3]))
        self.assertTrue(torch.equal(hybrid[:, 3:21], target[:, 3:21]))
        self.assertTrue(torch.equal(hybrid[:, 21:], source[:, 21:]))

    def test_topology_adherence_detects_spatial_cell_escape(self) -> None:
        spec = GridSpec((2, 2, 2), 4)
        values = _features()
        assigned = torch.tensor([0, 6])
        matched = topology_adherence(
            values,
            assigned,
            spec=spec,
            total_frames=32,
            frontier=8,
            stride_frames=8,
            support_sigma=3.0,
        )
        shifted = values.clone()
        shifted[:, 0] *= -1
        escaped = topology_adherence(
            shifted,
            assigned,
            spec=spec,
            total_frames=32,
            frontier=8,
            stride_frames=8,
            support_sigma=3.0,
        )
        self.assertEqual(matched.spatial_cell_fraction, 1.0)
        self.assertEqual(escaped.spatial_cell_fraction, 0.0)

    def test_render_signature_reports_lost_detail(self) -> None:
        target = torch.zeros(3, 4, 4, 3)
        target[:, ::2, ::2] = 1.0
        blurred = target.mean(dim=(1, 2), keepdim=True).expand_as(target)
        signature = render_signature(blurred, target)
        self.assertLess(signature.contrast_ratio, 0.01)
        self.assertLess(signature.edge_ratio, 0.01)
        self.assertTrue(math.isfinite(signature.psnr))


if __name__ == "__main__":
    unittest.main()
