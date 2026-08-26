"""Tests for prompt-caster data and control metrics."""

from __future__ import annotations

import unittest

import torch

from sol.prompt_jewel_caster import PromptJewelCaster
from sol.train_prompt_jewel_caster import (
    PromptSampleBatch,
    control_metrics,
    prompt_label,
)


class PromptCasterTrainingTests(unittest.TestCase):
    def test_prompt_label_includes_style_and_source_sentence(self) -> None:
        self.assertEqual(
            prompt_label({"style": "anime", "source_prompt": "a dancer spins"}),
            "anime video. a dancer spins",
        )

    def test_control_metrics_cover_three_arms_and_roles(self) -> None:
        model = PromptJewelCaster(
            text_dim=8, vocabulary_size=4, hidden_dim=16,
            depth=1, mixture_components=3,
        )
        batch = PromptSampleBatch(
            centers=torch.rand(13, 3) * 2 - 1,
            tokens=torch.randint(0, 4, (13, 3)),
            prompts=torch.arange(13) % 3,
        )
        embeddings = torch.randn(3, 8)
        report = control_metrics(
            model, batch, embeddings, torch.zeros(1, 8), chunk=5
        )
        self.assertEqual(set(report), {"correct", "shuffled", "null"})
        self.assertEqual(
            set(report["correct"]["token_nll"]),
            {"covariance", "surface", "gradient"},
        )


if __name__ == "__main__":
    unittest.main()
