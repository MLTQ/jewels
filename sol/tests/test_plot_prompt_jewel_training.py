"""Tests for neural prompt-speaker training-curve loading."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sol.plot_prompt_jewel_training import load_curve


class PromptTrainingPlotTests(unittest.TestCase):
    def test_all_controls_and_steps_are_preserved(self) -> None:
        report = {
            "schema": "prompt-native-jewel-caster-gate-v1",
            "best_step": 500,
            "history": [
                {
                    "step": step,
                    "controls": {
                        arm: {"token_nll_macro": value + offset, "centroid_nll": value}
                        for offset, arm in enumerate(("correct", "shuffled", "null"))
                    },
                }
                for step, value in ((500, 2.0), (1000, 3.0))
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report))
            curve = load_curve(path, report["schema"], "centroid_nll")
        self.assertEqual(curve["steps"], [500, 1000])
        self.assertEqual(curve["best_step"], 500)
        self.assertEqual(curve["token"]["shuffled"], [3.0, 4.0])

    def test_empty_history_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({"schema": "expected", "history": []}))
            with self.assertRaisesRegex(ValueError, "empty"):
                load_curve(path, "expected", "density_nce")


if __name__ == "__main__":
    unittest.main()
