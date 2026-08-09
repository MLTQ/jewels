"""Prompt-conditioned direct jewel evaluation tests."""

from __future__ import annotations

import math
import unittest

import torch

from sol.prompted_streaming_eval import evaluate_prompted_streaming
from sol.streaming_corpus import PromptedField, build_prompted_continuation_corpus
from sol.streaming_model import BirthContinuationModel
from sol.token_grid import GridSpec


def _features(color: float) -> torch.Tensor:
    features = torch.zeros(24, 22)
    features[:, 0] = torch.linspace(-0.8, 0.8, 24)
    features[:, 1] = torch.linspace(0.8, -0.8, 24)
    features[:, 2] = torch.linspace(-0.9, 0.9, 24)
    features[:, 3] = 2 * math.log(0.08)
    features[:, 6] = 2 * math.log(0.08)
    features[:, 8] = 2 * math.log(0.16)
    features[:, 9:12] = color
    features[:, 21] = 2.0
    return features


class PromptedStreamingEvaluationTests(unittest.TestCase):
    def test_reports_full_context_and_text_only_controls(self) -> None:
        fields = []
        for class_id in range(2):
            fields.extend(
                [
                    PromptedField(
                        f"class{class_id}_train",
                        class_id,
                        f"class{class_id}",
                        "train",
                        _features(0.2 + class_id),
                        32,
                        (class_id,),
                        (class_id + 2,),
                    ),
                    PromptedField(
                        f"class{class_id}_validation",
                        class_id,
                        f"class{class_id}",
                        "validation",
                        _features(0.3 + class_id),
                        32,
                        (class_id,),
                        (class_id + 2,),
                    ),
                ]
            )
        embeddings = torch.eye(4)
        spec = GridSpec((4, 4, 2), 8)
        corpus = build_prompted_continuation_corpus(
            fields,
            embeddings,
            prefix_frames=8,
            stride_frames=4,
            support_sigma=2.0,
            grid_spec=spec,
        )
        model = BirthContinuationModel(
            model_dim=32,
            grid_spec=spec,
            context_depth=1,
            cell_depth=1,
            slot_depth=1,
            context_mode="local",
            text_dim=4,
        )
        report = evaluate_prompted_streaming(model, corpus, device="cpu")
        self.assertGreater(report.validation_views, 0)
        self.assertEqual(set(report.full_context), {"correct", "shuffled", "null"})
        self.assertEqual(set(report.text_only), {"correct", "shuffled", "null"})
        for family in (report.full_context, report.text_only):
            for metric in family.values():
                self.assertTrue(math.isfinite(metric.feature_mse))
                self.assertTrue(math.isfinite(metric.count_mae))


if __name__ == "__main__":
    unittest.main()
