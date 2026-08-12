"""Tests for fixed-path initial/continuation scaffold mark controls."""

from __future__ import annotations

import unittest

import torch

from sol.birth_mark_flow import BirthMarkFlowModel
from sol.scaffold_mark_data import build_scaffold_mark_corpus
from sol.scaffold_mark_eval import evaluate_scaffold_mark_flow
from sol.streaming_corpus import PromptedField
from sol.token_grid import GridSpec


def _field(source_id: str, class_id: int, split: str) -> PromptedField:
    features = torch.zeros(12, 22)
    features[:, :3] = torch.rand(12, 3) * 1.6 - 0.8
    features[:, 2] = torch.linspace(-0.9, 0.9, 12)
    features[:, 3] = -5.0
    features[:, 6] = -5.0
    features[:, 8] = -3.0
    features[:, 9:12] = 0.5
    features[:, 21] = 2.0
    return PromptedField(
        source_id,
        class_id,
        f"class-{class_id}",
        split,
        features,
        16,
        (class_id,),
        (class_id + 2,),
    )


class ScaffoldMarkEvalTests(unittest.TestCase):
    def test_reports_initial_and_continuation_controls(self) -> None:
        torch.manual_seed(4)
        spec = GridSpec((2, 2, 2), 8)
        corpus = build_scaffold_mark_corpus(
            [
                _field("train-a", 0, "train"),
                _field("train-b", 1, "train"),
                _field("valid-a", 0, "validation"),
                _field("valid-b", 1, "validation"),
            ],
            torch.randn(4, 8),
            stride_frames=8,
            support_sigma=2.0,
            grid_spec=spec,
        )
        model = BirthMarkFlowModel(
            model_dim=8,
            grid_spec=spec,
            context_depth=1,
            noisy_depth=1,
            guide_depth=1,
            cell_depth=1,
            mark_depth=1,
            text_dim=8,
            guide_dim=3,
            guide_heads=1,
        )
        guides = {
            (source.field.source_id, view.index): torch.rand(spec.n_cells, 3)
            for source in corpus.sources
            for view in source.views
        }
        report = evaluate_scaffold_mark_flow(
            model, corpus, guides, device="cpu", seed=9
        )
        self.assertEqual(report["validation_views"], 4)
        for section in ("aggregate", "initial", "continuation"):
            self.assertTrue(torch.isfinite(torch.tensor(list(report[section].values()))).all())


if __name__ == "__main__":
    unittest.main()
