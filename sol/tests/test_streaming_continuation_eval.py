"""Tests for continuation selectivity and rendered-field evaluation plumbing."""

from __future__ import annotations

import math
import unittest

import torch

from sol.streaming_continuation_eval import (
    _nonoverlapping_context_index,
    evaluate_continuation,
)
from sol.streaming_data import build_continuation_dataset
from sol.streaming_model import BirthContinuationModel
from sol.token_grid import GridSpec


def _features() -> torch.Tensor:
    features = torch.zeros(16, 22)
    features[:, 0] = torch.linspace(-0.8, 0.8, 16)
    features[:, 1] = torch.linspace(0.8, -0.8, 16)
    features[:, 2] = torch.linspace(-0.9, 0.9, 16)
    features[:, 3] = 2 * math.log(0.1)
    features[:, 6] = 2 * math.log(0.1)
    features[:, 8] = 2 * math.log(0.18)
    features[:, 9:12] = 0.2
    features[:, 21] = 1.0
    return features


class StreamingContinuationEvaluationTests(unittest.TestCase):
    def test_shuffled_context_does_not_overlap_target_stride(self) -> None:
        spec = GridSpec((2, 2, 2), 8)
        dataset = build_continuation_dataset(
            _features(),
            24,
            prefix_frames=8,
            stride_frames=4,
            support_sigma=2.0,
            grid_spec=spec,
        )
        for index, view in enumerate(dataset.views):
            shuffled = dataset.views[
                _nonoverlapping_context_index(
                    dataset.views, index, dataset.prefix_frames
                )
            ]
            context_interval = (
                shuffled.frontier - dataset.prefix_frames,
                shuffled.frontier,
            )
            target_interval = (view.frontier, view.commit_stop)
            overlap = max(context_interval[0], target_interval[0]) < min(
                context_interval[1], target_interval[1]
            )
            self.assertFalse(overlap)

    def test_evaluation_is_finite_and_carry_is_exact(self) -> None:
        spec = GridSpec((2, 2, 2), 8)
        dataset = build_continuation_dataset(
            _features(),
            24,
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
        )
        evaluation = evaluate_continuation(
            model, dataset, device="cpu", points_per_frame=1
        )
        self.assertEqual(evaluation.carried_max_error, 0.0)
        self.assertTrue(
            all(torch.isfinite(torch.tensor(value)) for value in evaluation.to_dict().values())
        )


if __name__ == "__main__":
    unittest.main()
