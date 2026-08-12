"""Tests for autonomous topology-realizer report aggregation."""

from __future__ import annotations

import unittest

from sol.render_scaffold_topology_realizer import _macro_average


class RenderScaffoldTopologyRealizerTests(unittest.TestCase):
    def test_macro_average_weights_sources_equally(self) -> None:
        records = [
            {"metrics": {"psnr": 10.0, "ssim": 0.4}},
            {"metrics": {"psnr": 20.0, "ssim": 0.8}},
        ]
        result = _macro_average(records, "metrics")
        self.assertAlmostEqual(result["psnr"], 15.0)
        self.assertAlmostEqual(result["ssim"], 0.6)

    def test_macro_average_rejects_misaligned_sections(self) -> None:
        with self.assertRaisesRegex(ValueError, "different keys"):
            _macro_average(
                [
                    {"metrics": {"psnr": 10.0}},
                    {"metrics": {"ssim": 0.8}},
                ],
                "metrics",
            )


if __name__ == "__main__":
    unittest.main()
