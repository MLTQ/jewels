"""Tests for factorized prompt control aggregation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sol.prompt_jewel_caster import FactorizedPromptJewelCaster
from sol.train_factorized_prompt_jewel_caster import (
    FactorPromptBatch,
    exact_prompt_source_counts,
    factor_control_metrics,
    select_prompt_splits,
)


class FactorizedPromptTrainingTests(unittest.TestCase):
    def test_controls_cover_density_and_every_active_role(self) -> None:
        model = FactorizedPromptJewelCaster(
            text_dim=8, vocabulary_size=4, hidden_dim=16, depth=1
        )
        batch = FactorPromptBatch(
            centers=torch.rand(13, 3) * 2 - 1,
            negative_centers=torch.rand(13, 3) * 2 - 1,
            tokens=torch.randint(0, 4, (13, 3)),
            combinations=torch.arange(13) % 3,
        )
        report = factor_control_metrics(
            model, batch, torch.randn(3, 8), torch.randn(3, 8), chunk=5
        )
        self.assertEqual(set(report), {"correct", "shuffled", "null"})
        self.assertIn("density_nce", report["correct"])
        self.assertEqual(
            set(report["correct"]["token_nll"]),
            {"covariance", "surface", "gradient"},
        )

    def test_exact_prompt_counts_require_style_and_action_text(self) -> None:
        training = [
            SimpleNamespace(path="train-exact", source_id="train-a"),
            SimpleNamespace(path="train-neighbor", source_id="train-b"),
            SimpleNamespace(path="train-wrong-style", source_id="train-c"),
        ]
        validation = [
            SimpleNamespace(path="validation-0", source_id="heldout"),
            SimpleNamespace(path="validation-1", source_id="heldout"),
        ]
        metadata = {
            "train-exact": {"style": "anime", "source_prompt": "pirouette"},
            "train-neighbor": {"style": "anime", "source_prompt": "stage leap"},
            "train-wrong-style": {"style": "cartoon", "source_prompt": "pirouette"},
            "validation-0": {"style": "anime", "source_prompt": "pirouette"},
            "validation-1": {"style": "anime", "source_prompt": "pirouette"},
        }
        self.assertEqual(
            exact_prompt_source_counts(training, validation, metadata),
            {"heldout": 1},
        )

    def test_validation_replica_prompt_mismatch_is_rejected(self) -> None:
        validation = [
            SimpleNamespace(path="a", source_id="heldout"),
            SimpleNamespace(path="b", source_id="heldout"),
        ]
        metadata = {
            "a": {"style": "anime", "source_prompt": "pirouette"},
            "b": {"style": "anime", "source_prompt": "stage leap"},
        }
        with self.assertRaisesRegex(ValueError, "disagree"):
            exact_prompt_source_counts([], validation, metadata)

    def test_training_source_filter_keeps_only_registered_sources(self) -> None:
        records = [
            SimpleNamespace(source_id="train-a"),
            SimpleNamespace(source_id="train-b"),
            SimpleNamespace(source_id="heldout"),
        ]
        training, validation = select_prompt_splits(
            records, {"heldout"}, {"train-b"}
        )
        self.assertEqual([record.source_id for record in training], ["train-b"])
        self.assertEqual([record.source_id for record in validation], ["heldout"])

    def test_training_source_filter_rejects_missing_registration(self) -> None:
        records = [
            SimpleNamespace(source_id="train-a"),
            SimpleNamespace(source_id="heldout"),
        ]
        with self.assertRaisesRegex(ValueError, "missing"):
            select_prompt_splits(records, {"heldout"}, {"train-missing"})


if __name__ == "__main__":
    unittest.main()
