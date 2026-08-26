"""Tests for prompt-only trajectory-speaker evaluation."""

from __future__ import annotations

import unittest
from pathlib import Path

from sol import audit_prompt_trajectory_speaker
from sol.audit_prompt_trajectory_speaker import semantic_summary


class PromptTrajectorySpeakerAuditTests(unittest.TestCase):
    def test_semantic_summary(self) -> None:
        records = []
        for scene in range(3):
            for seed in range(3):
                records.append({
                    "scene_token": scene,
                    "correct_top1": seed != 2,
                    "correct_similarity": 0.40,
                    "shuffled_generation_similarity": 0.30,
                    "null_generation_similarity": 0.32,
                })
        summary = semantic_summary(records)
        self.assertEqual(summary["correct_top1"], 6)
        self.assertEqual(summary["classes_with_majority_retrieval"], 3)
        self.assertAlmostEqual(summary["correct_minus_shuffled_generation"], 0.10)
        self.assertEqual(summary["correct_beats_shuffled_generation"], 9)

    def test_entry_point_and_protocol_exist(self) -> None:
        self.assertTrue(callable(audit_prompt_trajectory_speaker.main))
        protocol = Path(__file__).parents[1] / "results" / "jewel_casting_language_v0" / (
            "PROTOCOL_PROMPT_TRAJECTORY_SPEAKER_V1.md"
        )
        self.assertTrue(protocol.exists())


if __name__ == "__main__":
    unittest.main()
