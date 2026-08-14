"""Tests for scaling-curve point collection and rendering."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from sol.render_scaling_curve import collect_point, render_curve


def _mark_summary(correct: float) -> dict:
    return {
        "latest_evaluation": {
            "aggregate": {
                "correct": correct,
                "shuffled_scaffold": correct + 0.03,
            }
        }
    }


def _report(psnr: float, lpips: float) -> dict:
    return {
        "macro": {
            "generated subset": {
                "psnr": psnr,
                "ssim": 0.5,
                "lpips_mean": lpips,
            }
        }
    }


class RenderScalingCurveTests(unittest.TestCase):
    def test_collect_point_extracts_metrics_and_margin(self) -> None:
        point = collect_point(
            4, _mark_summary(1.2), _report(13.0, 0.5), "generated subset"
        )
        self.assertEqual(point["sources"], 4)
        self.assertAlmostEqual(point["shuffled_minus_correct"], 0.03)
        self.assertEqual(point["rollout_psnr"], 13.0)

    def test_collect_point_rejects_unknown_arm(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks arm"):
            collect_point(4, _mark_summary(1.2), _report(13.0, 0.5), "missing")

    def test_render_curve_writes_figure(self) -> None:
        points = [
            collect_point(
                n, _mark_summary(1.3 - 0.05 * n), _report(12 + n / 4, 0.6 - n / 100),
                "generated subset",
            )
            for n in (4, 8, 12)
        ]
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "curve.png"
            render_curve(points, out)
            self.assertGreater(out.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
