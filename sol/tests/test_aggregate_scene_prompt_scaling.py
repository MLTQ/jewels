"""Tests for shared-scene exact-prompt data scaling aggregation."""

from __future__ import annotations

import unittest

from sol.aggregate_scene_prompt_scaling import aggregate


def _report(count: int, token_margin: float, histogram_margin: float, retrieval: int, passed: bool) -> dict:
    correct_token = 7.0
    correct_histogram = 0.15
    return {
        "protocol": {"training_fields": count * 3},
        "best_step": 500,
        "teacher_forced_controls": {
            "correct": {"token_nll_macro": correct_token, "density_nce": 0.6},
            "shuffled": {"token_nll_macro": 7.3, "density_nce": 0.62},
            "null": {"token_nll_macro": correct_token + token_margin, "density_nce": 0.61},
        },
        "generation_macro": {
            "correct": {"target_histogram_cosine": correct_histogram},
            "shuffled": {"target_histogram_cosine": 0.1},
            "null": {"target_histogram_cosine": correct_histogram - histogram_margin},
        },
        "retrieval": [{"correct": index < retrieval} for index in range(3)],
        "gate": {"passed": passed},
    }


class ScenePromptScalingTests(unittest.TestCase):
    def test_positive_curve_requires_every_registered_signal(self) -> None:
        report = aggregate([
            (2, _report(2, -0.1, 0.0, 1, False)),
            (4, _report(4, 0.0, 0.01, 2, False)),
            (6, _report(6, 0.1, 0.03, 3, True)),
        ])
        self.assertTrue(report["positive_data_scaling"])
        self.assertTrue(report["final_absolute_gate_passed"])

    def test_nonmonotonic_metric_keeps_curve_negative(self) -> None:
        report = aggregate([
            (2, _report(2, 0.0, 0.01, 2, False)),
            (4, _report(4, 0.1, 0.00, 2, False)),
        ])
        self.assertFalse(report["positive_data_scaling"])


if __name__ == "__main__":
    unittest.main()
