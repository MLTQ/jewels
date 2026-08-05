"""Tests for multi-clip prompted continuation corpus construction."""

from __future__ import annotations

import math
import unittest

import torch

from sol.streaming_corpus import PromptedField, build_prompted_continuation_corpus
from sol.streaming_data import FeatureStandardizer, build_continuation_dataset
from sol.token_grid import GridSpec


def _features(color: float) -> torch.Tensor:
    centers = torch.linspace(-0.9, 0.9, 24)
    features = torch.zeros(len(centers), 22)
    features[:, 0] = torch.linspace(-0.8, 0.8, len(centers))
    features[:, 1] = torch.linspace(0.8, -0.8, len(centers))
    features[:, 2] = centers
    features[:, 3] = 2 * math.log(0.08)
    features[:, 6] = 2 * math.log(0.08)
    features[:, 8] = 2 * math.log(0.16)
    features[:, 9:12] = color
    features[:, 21] = 2.0
    return features


class StreamingCorpusTests(unittest.TestCase):
    def test_shared_normalization_uses_train_sources_only(self) -> None:
        fields = [
            PromptedField(
                "class0_train",
                0,
                "class0",
                "train",
                _features(0.2),
                32,
                (0,),
                (1,),
            ),
            PromptedField(
                "class0_validation",
                0,
                "class0",
                "validation",
                _features(50.0),
                32,
                (0,),
                (1,),
            ),
        ]
        spec = GridSpec((4, 4, 2), 8)
        corpus = build_prompted_continuation_corpus(
            fields,
            torch.eye(2),
            prefix_frames=8,
            stride_frames=4,
            support_sigma=2.0,
            grid_spec=spec,
        )
        train_only = build_continuation_dataset(
            fields[0].features,
            32,
            prefix_frames=8,
            stride_frames=4,
            support_sigma=2.0,
            grid_spec=spec,
        )
        expected_context = FeatureStandardizer.fit(
            [view.context_features for view in train_only.views]
        )
        expected_birth = FeatureStandardizer.fit(
            [view.births.values for view in train_only.views]
        )
        self.assertTrue(
            torch.equal(corpus.context_standardizer.mean, expected_context.mean)
        )
        self.assertTrue(torch.equal(corpus.birth_standardizer.mean, expected_birth.mean))
        self.assertIs(
            corpus.train[0].dataset.context_standardizer,
            corpus.validation[0].dataset.context_standardizer,
        )

    def test_requires_every_class_in_both_splits(self) -> None:
        fields = [
            PromptedField("a", 0, "a", "train", _features(0.1), 32, (0,), (1,)),
            PromptedField(
                "b", 1, "b", "validation", _features(0.2), 32, (2,), (3,)
            ),
        ]
        with self.assertRaises(ValueError):
            build_prompted_continuation_corpus(
                fields,
                torch.eye(4),
                prefix_frames=8,
                stride_frames=4,
                support_sigma=2.0,
                grid_spec=GridSpec((4, 4, 2), 8),
            )


if __name__ == "__main__":
    unittest.main()
