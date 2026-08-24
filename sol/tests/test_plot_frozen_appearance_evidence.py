"""Tests for frozen-appearance evidence collection and plotting."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sol.plot_frozen_appearance_evidence import collect_evidence, plot_evidence


def _report(macros: dict[str, tuple[float, float]]) -> dict:
    records = []
    for arm in macros:
        records.append({
            "arm": arm,
            "render_signature": {"temporal_change_ratio": 1.5},
        })
    return {
        "perceptual_macro": {
            arm: {"psnr": values[0], "lpips": values[1]}
            for arm, values in macros.items()
        },
        "perceptual_records": records,
    }


class FrozenAppearanceEvidenceTests(unittest.TestCase):
    def test_collects_registered_arm_mapping_and_writes_figure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "audit_seed0").mkdir()
            (root / "replication" / "audit_final_seeds").mkdir(parents=True)
            (root / "screens").mkdir()
            ablation = _report({
                "irregular_seed0": (19.6, 0.73),
                "irregular_seed1": (19.9, 0.72),
                "irregular_seed2": (19.8, 0.725),
                "irregular_seed3": (19.79, 0.726),
            })
            replication = _report({
                "irregular_seed1": (20.08, 0.721),
                "irregular_seed2": (20.03, 0.722),
                "irregular_seed3": (20.05, 0.723),
            })
            (root / "audit_seed0" / "report.json").write_text(json.dumps(ablation))
            (root / "replication" / "audit_final_seeds" / "report.json").write_text(
                json.dumps(replication)
            )
            for name, fraction in (
                ("frozen_render_seed0_600", 0.04),
                ("frozen_perceptual_seed0_600", 0.035),
                ("frozen_stabilized_seed0_600", 0.03),
            ):
                path = root / "screens" / name
                path.mkdir()
                (path / "train_log.jsonl").write_text(json.dumps({
                    "render_loss": 0.1,
                    "render_out_of_range_fraction": fraction,
                }) + "\n")
            evidence = collect_evidence(root)
            self.assertEqual(evidence["compute"]["psnr"][-1], 20.08)
            self.assertLess(evidence["ablation_delta"]["psnr"][0], 0)
            output = root / "evidence.png"
            plot_evidence(evidence, output)
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
