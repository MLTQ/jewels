"""Tests for appearance-contract replication parsing."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sol.plot_appearance_replication import load_replication


class AppearanceReplicationPlotTests(unittest.TestCase):
    def test_load_replication_preserves_seed_order_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for seed in range(3):
                path = Path(directory) / f"seed{seed}.json"
                path.write_text(json.dumps({
                    "latest_evaluation": {
                        "macro_psnr": 19.0 + seed / 10,
                        "structure": {
                            "occupancy_uniformity": 0.984 + seed / 1000,
                            "active_fraction": 0.62 + seed / 100,
                            "mixed_spacetime_tilt_median": 0.52 + seed / 100,
                        },
                    }
                }))
                paths.append(path)
            rows = load_replication(paths)
            self.assertEqual(rows[0]["psnr"], 19.0)
            self.assertAlmostEqual(rows[2]["occupancy"], 0.986)
            self.assertAlmostEqual(rows[1]["tilt"], 0.53)


if __name__ == "__main__":
    unittest.main()
