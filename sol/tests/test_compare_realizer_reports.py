"""Matched realizer-report comparison tests."""

from __future__ import annotations

import unittest

from sol.compare_realizer_reports import compare_realizer_reports


def _record(source: str, psnr: float, ssim: float) -> dict:
    render = {
        "psnr": psnr,
        "ssim": ssim,
        "contrast_ratio": 0.8,
        "edge_ratio": 0.9,
        "saturation_ratio": 1.0,
        "temporal_change_ratio": 1.1,
    }
    topology = {
        "spatial_cell_fraction": 1.0,
        "birth_cell_fraction": 0.99,
        "birth_commit_fraction": 1.0,
    }
    return {
        "source_id": source,
        "class_name": source,
        "render_signatures": {"flow guided": render},
        "topology_adherence": {"flow guided": topology},
    }


class CompareRealizerReportsTests(unittest.TestCase):
    def test_reports_macro_and_paired_source_deltas(self) -> None:
        baseline = [_record("b", 10.0, 0.5), _record("a", 20.0, 0.7)]
        candidate = [_record("a", 22.0, 0.8), _record("b", 9.0, 0.4)]
        report = compare_realizer_reports(baseline, candidate)
        self.assertEqual(report["sources"], ["a", "b"])
        self.assertEqual(report["baseline"]["render"]["psnr"], 15.0)
        self.assertEqual(report["candidate"]["render"]["psnr"], 15.5)
        self.assertEqual(report["delta"]["render"]["psnr"], 0.5)
        self.assertAlmostEqual(report["per_source"][0]["delta"]["ssim"], 0.1)

    def test_rejects_unmatched_sources(self) -> None:
        with self.assertRaises(ValueError):
            compare_realizer_reports([_record("a", 1, 1)], [_record("b", 1, 1)])

    def test_rejects_missing_panel(self) -> None:
        baseline = [_record("a", 1, 1)]
        candidate = [_record("a", 1, 1)]
        candidate[0]["render_signatures"] = {}
        with self.assertRaises(ValueError):
            compare_realizer_reports(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
