"""Tests for Gate 2a block-oracle training helpers."""

from __future__ import annotations

import unittest
from argparse import Namespace
import json
from pathlib import Path
import tempfile

import torch

from sol.audit_jewel_casting_language import FieldRecord
from sol.block_token_jewel_speaker import BlockTokenJewelSpeaker
from sol.train_block_token_oracle import (
    BlockOracleBatch,
    cyclic_shuffled_programs,
    oracle_control_metrics,
    resolve_split_arguments,
)


def _record(source: str, fit_seed: int) -> FieldRecord:
    return FieldRecord(
        path=f"/{source}/{fit_seed}.pt", source_id=source,
        style="style", fit_seed=fit_seed,
        features=torch.zeros(2, 22), background=torch.zeros(3),
    )


class BlockTokenOracleTrainingTests(unittest.TestCase):
    def test_shuffled_programs_preserve_fit_rank(self) -> None:
        records = [
            _record("a", 1), _record("a", 2),
            _record("b", 1), _record("b", 2),
            _record("c", 1), _record("c", 2),
        ]
        programs = torch.arange(6)[:, None].expand(6, 4)
        shuffled = cyclic_shuffled_programs(records, programs)
        self.assertEqual(shuffled[:, 0].tolist(), [2, 3, 4, 5, 0, 1])

    def test_controls_cover_all_arms_and_roles(self) -> None:
        model = BlockTokenJewelSpeaker(
            block_vocabulary_size=5, jewel_vocabulary_size=7,
            block_shape=(2, 2, 1), hidden_dim=16, depth=1,
        )
        batch = BlockOracleBatch(
            centers=torch.rand(17, 3) * 2 - 1,
            negative_centers=torch.rand(17, 3) * 2 - 1,
            jewel_tokens=torch.randint(0, 7, (17, 3)),
            owners=torch.arange(17) % 3,
        )
        programs = torch.randint(0, 5, (3, 4))
        report = oracle_control_metrics(
            model, batch, programs, torch.roll(programs, -1, 0), 0, chunk=6
        )
        self.assertEqual(set(report), {"oracle block", "shuffled block", "null block"})
        self.assertEqual(
            set(report["oracle block"]["token_nll"]),
            {"covariance", "surface", "gradient"},
        )
        self.assertIn("density_nce", report["null block"])

    def test_program_alignment_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "align"):
            cyclic_shuffled_programs([_record("a", 1), _record("b", 1)], torch.zeros(1, 4))

    def test_split_can_be_owned_by_prior_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text(json.dumps({"protocol": {
                "roots": ["r"], "validation_sources": ["v"],
                "training_sources": ["t"],
            }}))
            resolved = resolve_split_arguments(Namespace(
                split_report=str(report), root=None,
                validation_source=None, training_source=None,
            ))
        self.assertEqual(resolved, (["r"], ["v"], ["t"]))


if __name__ == "__main__":
    unittest.main()
