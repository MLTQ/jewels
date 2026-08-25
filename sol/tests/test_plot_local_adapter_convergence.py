"""Tests for local-adapter convergence evidence collection."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sol.plot_local_adapter_convergence import collect_convergence


def _write_report(path: Path, lpips: list[float], psnr: list[float]) -> None:
    path.mkdir(parents=True)
    macro = {
        f"irregular_seed{index}": {"lpips": l_value, "psnr": p_value}
        for index, (l_value, p_value) in enumerate(zip(lpips, psnr, strict=True))
    }
    (path / "report.json").write_text(json.dumps({"perceptual_macro": macro}))


def _write_log(path: Path, step: int, lpips: float, psnr: float) -> None:
    path.mkdir(parents=True)
    rows = [
        {
            "step": step,
            "appearance_lpips": lpips,
            "render_loss": 0.01,
            "render_out_of_range_fraction": 0.03,
        },
        {"step": step, "evaluation": {"macro_psnr": psnr}},
    ]
    (path / "train_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


class LocalAdapterConvergencePlotTests(unittest.TestCase):
    def test_collect_convergence_offsets_continuation_and_maps_audits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            screen = root / "screen"
            convergence = root / "convergence"
            _write_report(
                screen / "audit_final_seed0_400",
                [0.72, 0.71, 0.70, 0.69],
                [20.0, 20.1, 20.2, 20.3],
            )
            _write_report(
                convergence / "audit_raw_curve_8000",
                [0.72, 0.705, 0.698],
                [20.0, 20.1, 20.2],
            )
            _write_report(
                convergence / "audit_raw_plateau_16000",
                [0.72, 0.696, 0.6958],
                [20.0, 20.2, 20.21],
            )
            _write_report(
                convergence / "audit_derivative_progress_seed0",
                [0.72, 0.71, 0.70, 0.69, 0.68, 0.67, 0.66, 0.659],
                [20.0, 20.1, 20.2, 20.3, 20.5, 20.7, 20.8, 20.83],
            )
            _write_report(
                convergence / "audit_derivative_replication_curve",
                [0.72, 0.67, 0.66, 0.659, 0.668, 0.661, 0.660],
                [20.0, 20.7, 20.8, 20.83, 20.71, 20.81, 20.82],
            )
            _write_log(
                convergence / "raw_r2_lpips005_seed0_12k", 12000, 0.12, 20.2
            )
            _write_log(
                convergence / "raw_r2_lpips005_seed0_continue4k", 4000, 0.11, 20.2
            )
            _write_log(
                convergence / "derivative32_r2_lpips005_seed0_12k",
                12000,
                0.10,
                20.9,
            )
            _write_log(
                convergence / "derivative32_r2_lpips005_seed1_12k",
                12000,
                0.10,
                20.8,
            )
            evidence = collect_convergence(screen, convergence)
            self.assertEqual(
                evidence["training"]["raw local"][-1]["step"], 16000
            )
            self.assertEqual(
                evidence["validation"]["raw local"][-1]["macro_psnr"], 20.2
            )
            self.assertEqual(
                evidence["exact"]["raw local"][-1]["lpips"], 0.6958
            )
            self.assertEqual(
                evidence["exact"]["derivative x32 seed 0"][-1]["step"],
                12000,
            )
            self.assertEqual(
                evidence["exact"]["derivative x32 seed 1"][-1]["lpips"],
                0.660,
            )


if __name__ == "__main__":
    unittest.main()
