"""Prompted jewel mark-flow evaluation tests."""

from __future__ import annotations

import math
import unittest

import torch

from sol.birth_mark_flow import BirthMarkFlowModel
from sol.prompted_mark_flow_eval import evaluate_prompted_mark_flow
from sol.streaming_corpus import PromptedField, build_prompted_continuation_corpus
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


class PromptedMarkFlowEvaluationTests(unittest.TestCase):
    def test_reports_fixed_path_prompt_controls(self) -> None:
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
        model = BirthMarkFlowModel(
            model_dim=32,
            grid_spec=spec,
            context_depth=1,
            noisy_depth=1,
            cell_depth=1,
            mark_depth=1,
            text_dim=4,
        )
        report = evaluate_prompted_mark_flow(model, corpus, device="cpu")
        self.assertGreater(report.validation_views, 0)
        for family in (report.full_context, report.text_only):
            self.assertTrue(math.isfinite(family.correct))
            self.assertTrue(math.isfinite(family.shuffled))
            self.assertTrue(math.isfinite(family.null))

    def test_accepts_complete_multiscale_guide_mapping(self) -> None:
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
        spec = GridSpec((4, 4, 2), 8)
        corpus = build_prompted_continuation_corpus(
            fields,
            torch.eye(4),
            prefix_frames=8,
            stride_frames=4,
            support_sigma=2.0,
            grid_spec=spec,
        )
        model = BirthMarkFlowModel(
            model_dim=32,
            grid_spec=spec,
            context_depth=1,
            noisy_depth=1,
            cell_depth=1,
            mark_depth=1,
            text_dim=4,
            guide_token_dim=16,
            guide_heads=4,
        )
        guides = {
            (example.source_id, view.index): torch.randn(spec.n_cells, 2, 16)
            for example in corpus.validation
            for view in example.dataset.views
        }
        report = evaluate_prompted_mark_flow(
            model, corpus, device="cpu", guide_tokens=guides
        )
        self.assertGreater(report.validation_views, 0)


if __name__ == "__main__":
    unittest.main()
