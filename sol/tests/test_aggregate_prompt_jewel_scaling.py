"""Tests for prompt-to-Jewel data-scaling aggregation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sol.aggregate_prompt_jewel_scaling import aggregate, summarize_point


def _report(training: int, offset: float, retrieval: int = 1) -> dict:
    correct = 5.0 - offset
    return {
        "schema": "additive-prompt-native-jewel-caster-gate-v1",
        "protocol": {"training_fields": training},
        "teacher_forced_controls": {
            "correct": {"cell_nll": correct, "token_nll_macro": correct + 1},
            "shuffled": {"cell_nll": 5.0, "token_nll_macro": 6.0},
            "null": {"cell_nll": 4.5, "token_nll_macro": 5.5},
        },
        "generation_macro": {
            "correct": {"target_histogram_cosine": 0.2 + offset},
            "shuffled": {"target_histogram_cosine": 0.15},
            "null": {"target_histogram_cosine": 0.18},
        },
        "retrieval": [
            {"correct": index < retrieval} for index in range(3)
        ],
        "gate": {"passed": offset >= 0.2},
    }


class PromptScalingAggregationTests(unittest.TestCase):
    def test_summarize_uses_control_minus_correct_margin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(_report(9, 0.1)))
            point = summarize_point(path, "n9")
        self.assertAlmostEqual(point["cell_margin_vs_shuffled"], 0.1)
        self.assertAlmostEqual(point["cell_margin_vs_null"], -0.4)
        self.assertAlmostEqual(point["retrieval_accuracy"], 1 / 3)

    def test_positive_curve_needs_every_frozen_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            points = []
            for training, offset, retrieval in ((9, 0.05, 1), (33, 0.1, 2), (57, 0.2, 3)):
                path = Path(directory) / f"{training}.json"
                path.write_text(json.dumps(_report(training, offset, retrieval)))
                points.append(summarize_point(path, f"n{training}"))
        result = aggregate(points)
        self.assertTrue(result["positive_data_scaling"])
        self.assertTrue(result["final_absolute_gate_passed"])

    def test_duplicate_training_count_is_rejected(self) -> None:
        point = {
            "training_fields": 9,
            "cell_margin_vs_shuffled": 0.0,
            "token_margin_vs_shuffled": 0.0,
            "target_histogram_cosine": {"correct": 0.0},
            "retrieval_accuracy": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "unique"):
            aggregate([point, dict(point)])


if __name__ == "__main__":
    unittest.main()
