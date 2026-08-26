"""Tests for exact-prompt source-repetition aggregation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sol.aggregate_prompt_repetition import aggregate, summarize


def _report(training: int, gain: float, retrieval: int, passed: bool) -> dict:
    return {
        "schema": "factorized-prompt-native-jewel-caster-gate-v1",
        "protocol": {"training_fields": training},
        "best_step": 500,
        "teacher_forced_controls": {
            "correct": {"density_nce": 1 - gain, "token_nll_macro": 2 - gain},
            "shuffled": {"density_nce": 1.0, "token_nll_macro": 2.0},
            "null": {"density_nce": 0.9, "token_nll_macro": 1.9},
        },
        "generation_macro": {
            "correct": {"target_histogram_cosine": 0.2 + gain},
            "shuffled": {"target_histogram_cosine": 0.15},
            "null": {"target_histogram_cosine": 0.18},
        },
        "retrieval": [{"correct": index < retrieval} for index in range(3)],
        "gate": {"passed": passed},
    }


class PromptRepetitionAggregationTests(unittest.TestCase):
    def test_control_minus_correct_margins_and_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(_report(60, 0.2, 2, True)))
            point = summarize(path, 1)
        self.assertAlmostEqual(point["density_margin_vs_shuffled"], 0.2)
        self.assertAlmostEqual(point["density_margin_vs_null"], 0.1)
        self.assertAlmostEqual(point["retrieval_accuracy"], 2 / 3)

    def test_positive_scaling_requires_every_registered_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            points = []
            for repetition, gain, retrieval in ((1, 0.1, 1), (2, 0.2, 2)):
                path = Path(directory) / f"{repetition}.json"
                path.write_text(json.dumps(_report(58 + repetition, gain, retrieval, repetition == 2)))
                points.append(summarize(path, repetition))
        result = aggregate(points)
        self.assertTrue(result["positive_repetition_scaling"])
        self.assertTrue(result["final_absolute_gate_passed"])


if __name__ == "__main__":
    unittest.main()
