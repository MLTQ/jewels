"""Tests for the corrected encoder convergence protocol and aggregation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sol.aggregate_encoder_convergence import confidence
from sol.run_encoder_convergence import validate_nested_manifests


def _manifest(train: list[str], validation: list[str]) -> dict:
    return {"examples": [
        *({"source_id": source_id, "split": "train"} for source_id in train),
        *({"source_id": source_id, "split": "validation"}
          for source_id in validation),
    ]}


class EncoderConvergenceProtocolTests(unittest.TestCase):
    def test_nested_training_and_frozen_validation_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for size, train in ((2, ["a", "b"]), (3, ["a", "b", "c"])):
                directory = root / f"n{size}"
                directory.mkdir()
                (directory / "manifest.json").write_text(
                    json.dumps(_manifest(train, ["v1", "v2"]))
                )
            report = validate_nested_manifests(root, [2, 3])
            self.assertEqual(report["validation_count"], 2)
            self.assertEqual(len(report["validation_sha256"]), 64)

    def test_nonprefix_training_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for size, train in ((2, ["a", "b"]), (3, ["a", "c", "b"])):
                directory = root / f"n{size}"
                directory.mkdir()
                (directory / "manifest.json").write_text(
                    json.dumps(_manifest(train, ["v"]))
                )
            with self.assertRaisesRegex(ValueError, "not a prefix"):
                validate_nested_manifests(root, [2, 3])

    def test_three_seed_interval_contains_mean(self) -> None:
        report = confidence([21.0, 22.0, 23.0])
        self.assertEqual(report["mean"], 22.0)
        self.assertLess(report["ci95_low"], report["mean"])
        self.assertGreater(report["ci95_high"], report["mean"])


if __name__ == "__main__":
    unittest.main()
