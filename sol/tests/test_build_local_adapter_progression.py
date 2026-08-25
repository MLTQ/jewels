"""Tests for pitch progression sheet construction."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from sol.build_local_adapter_progression import build_progression


class LocalAdapterProgressionTests(unittest.TestCase):
    def test_build_progression_drops_lattice_and_writes_style_strips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "audit"
            audit.mkdir()
            report = {
                "protocol": {"candidate_labels": ["source", "400 updates"]},
                "perceptual_macro": {
                    "irregular_seed0": {"lpips": 0.72, "psnr": 20.0},
                    "irregular_seed1": {"lpips": 0.70, "psnr": 20.2},
                },
                "perceptual_records": [
                    {"style": "anime"},
                    {"style": "clay"},
                ],
            }
            (audit / "report.json").write_text(json.dumps(report))
            # Five columns: target, lattice, two candidates, teacher; two rows.
            Image.new("RGB", (50, 16), "red").save(audit / "qualitative.png")
            out = root / "pitch.png"
            style_dir = root / "styles"
            evidence = build_progression(audit, out, style_dir)
            self.assertEqual(evidence["output_width"], 40)
            with Image.open(style_dir / "anime.png") as anime:
                self.assertEqual(anime.size, (40, 8))
            self.assertTrue((style_dir / "clay.png").exists())
            with Image.open(out) as pitch:
                self.assertGreater(pitch.height, 16)


if __name__ == "__main__":
    unittest.main()
