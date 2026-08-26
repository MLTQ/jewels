"""Tests for learned trajectory-speaker training utilities."""

from __future__ import annotations

import unittest

import torch

from sol.learned_trajectory_speaker import LearnedTrajectorySpeaker
from sol.train_learned_trajectory_speaker import (
    PROMPT_PARAPHRASES,
    build_program_examples,
    evaluate_conditions,
)


class LearnedTrajectorySpeakerTrainingTests(unittest.TestCase):
    def test_program_pair_split(self) -> None:
        prompts = tuple(PROMPT_PARAPHRASES)
        sources = ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11),
                   (12, 13, 14, 15, 16, 17))
        training, evaluation = build_program_examples(prompts, sources)
        self.assertEqual(len(training), 216)
        self.assertEqual(len(evaluation), 18)
        train_pairs = {
            (row["scene_token"], row["foreground_token"], row["background_token"])
            for row in training
        }
        eval_pairs = {
            (row["scene_token"], row["foreground_token"], row["background_token"])
            for row in evaluation
        }
        self.assertFalse(train_pairs & eval_pairs)

    def test_condition_evaluation(self) -> None:
        prompts = tuple(PROMPT_PARAPHRASES)
        sources = ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11),
                   (12, 13, 14, 15, 16, 17))
        _, evaluation = build_program_examples(prompts, sources)
        all_prompts = {row["prompt"] for row in evaluation} | {""}
        embeddings = {
            prompt: torch.nn.functional.normalize(torch.randn(8), dim=0)
            for prompt in all_prompts
        }
        model = LearnedTrajectorySpeaker(8, 16, 3, 18)
        conditions = evaluate_conditions(model, evaluation, embeddings, prompts)
        self.assertEqual(set(conditions), {"correct", "shuffled", "null"})
        self.assertTrue(all(
            torch.isfinite(torch.tensor(row["token_nll_macro"]))
            for row in conditions.values()
        ))


if __name__ == "__main__":
    unittest.main()
