"""Tests for temporal-preserving spatial jewel splits."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch


STPRIM_ROOT = Path(__file__).resolve().parents[2] / "stprim"
if str(STPRIM_ROOT) not in sys.path:
    sys.path.insert(0, str(STPRIM_ROOT))

from core.params import PrimitiveField  # noqa: E402
from fit.adapt import GradientTracker, adapt  # noqa: E402


class SpatialDensificationTests(unittest.TestCase):
    def test_identity_split_preserves_time_and_total_volume(self) -> None:
        field = PrimitiveField(1, p1_color=True)
        with torch.no_grad():
            field.mu.zero_()
            field.log_scale.zero_()
            field.quat.zero_()
            field.quat[:, 0] = 1
            field.logit_w.zero_()
        tracker = GradientTracker(1, "cpu")
        tracker.accum.fill_(1)
        tracker.count = 1

        stats = adapt(
            field,
            tracker,
            max_primitives=2,
            densify_frac=1.0,
            split_mode="spatial",
            generator=torch.Generator().manual_seed(3),
        )

        self.assertEqual(stats["n"], 2)
        expected = torch.tensor([1 / math.sqrt(2), 1 / math.sqrt(2), 1.0])
        torch.testing.assert_close(field.scales(), expected.repeat(2, 1))
        torch.testing.assert_close(field.mu[:, 2], torch.zeros(2))
        self.assertAlmostEqual(
            float(field.scales().prod(dim=1).sum().detach()), 1.0, places=6
        )

    def test_rejects_unknown_split_mode(self) -> None:
        field = PrimitiveField(1)
        with self.assertRaises(ValueError):
            adapt(
                field,
                GradientTracker(1, "cpu"),
                max_primitives=2,
                split_mode="temporal-mystery",
            )


if __name__ == "__main__":
    unittest.main()
