"""Tests for the shared-scene causal evidence extractor."""

from __future__ import annotations

import unittest

from sol.plot_prompt_scene_ablation import extract_ablation


def _report(correct_hist: float, null_hist: float, passed: bool) -> dict:
    controls = {
        arm: {
            "density_nce": 0.6 + index * 0.01,
            "token_nll_macro": 7.0 + index * 0.1,
        }
        for index, arm in enumerate(("correct", "shuffled", "null"))
    }
    generation = {
        "correct": {"target_histogram_cosine": correct_hist},
        "shuffled": {"target_histogram_cosine": 0.1},
        "null": {"target_histogram_cosine": null_hist},
    }
    return {
        "teacher_forced_controls": controls,
        "generation_macro": generation,
        "retrieval": [{"correct": True}, {"correct": False}],
        "gate": {"passed": passed},
    }


class PromptSceneAblationTests(unittest.TestCase):
    def test_extracts_margin_reversal_without_turning_it_into_a_pass(self) -> None:
        report = extract_ablation(
            _report(0.14, 0.15, False), _report(0.16, 0.15, False)
        )
        self.assertAlmostEqual(report["rows"][0]["histogram_margin_vs_null"], -0.01)
        self.assertAlmostEqual(report["rows"][1]["histogram_margin_vs_null"], 0.01)
        self.assertTrue(report["shared_scene_improved_histogram_margin"])
        self.assertFalse(report["final_gate_passed"])


if __name__ == "__main__":
    unittest.main()
