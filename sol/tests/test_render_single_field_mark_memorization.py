"""Tests for the controlled single-field mark memorization renderer."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from sol.render_single_field_mark_memorization import (
    _single_field_identity,
    _teacher_forced_density,
)


class SingleFieldMarkMemorizationTests(unittest.TestCase):
    def test_identity_requires_one_shared_physical_field(self) -> None:
        source = SimpleNamespace(field=SimpleNamespace(source_id="valid-one"))
        corpus = SimpleNamespace(train=(object(),), validation=(source,))
        manifest = {
            "single_field_overfit_class": "PlayingGuitar",
            "examples": [
                {
                    "source_id": "train-one",
                    "split": "train",
                    "shared_field_stem": "field-one",
                },
                {
                    "source_id": "valid-one",
                    "split": "validation",
                    "shared_field_stem": "field-one",
                },
            ],
        }
        selected, item = _single_field_identity(manifest, corpus)
        self.assertIs(selected, source)
        self.assertEqual(item["source_id"], "valid-one")
        manifest["examples"][1]["shared_field_stem"] = "field-two"
        with self.assertRaisesRegex(ValueError, "physical field stem"):
            _single_field_identity(manifest, corpus)

    def test_density_stitches_only_each_committed_stride(self) -> None:
        field = torch.zeros(1, 22)
        field[:, 3:9] = torch.tensor((-4.0, 0.0, 0.0, -4.0, 0.0, -4.0))
        field[:, 21] = 4.0
        report = _teacher_forced_density(
            [field, field], total_frames=4, stride_frames=2, support_sigma=3.0
        )
        self.assertEqual(len(report["per_frame_effective"]), 4)
        self.assertEqual(len(report["per_frame_above_5_percent_alpha"]), 4)


if __name__ == "__main__":
    unittest.main()
