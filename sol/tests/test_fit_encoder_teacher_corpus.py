"""Tests for source-owned encoder teacher selection."""

from __future__ import annotations

import unittest

from sol.fit_encoder_teacher_corpus import safe_name, select_examples


class TeacherCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {"examples": [
            {"source_id": "a/x", "split": "train"},
            {"source_id": "b", "split": "train"},
            {"source_id": "v", "split": "validation"},
        ]}

    def test_ordered_prefix_selection(self) -> None:
        selected = select_examples(
            self.manifest, split="train", offset=0, limit=1, source_ids=()
        )
        self.assertEqual([row["source_id"] for row in selected], ["a/x"])

    def test_explicit_selection_preserves_requested_order(self) -> None:
        selected = select_examples(
            self.manifest, split="train", offset=0, limit=0, source_ids=("b", "a/x")
        )
        self.assertEqual([row["source_id"] for row in selected], ["b", "a/x"])

    def test_missing_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not in split"):
            select_examples(
                self.manifest, split="train", offset=0, limit=0, source_ids=("missing",)
            )

    def test_offset_selects_a_disjoint_ordered_slice(self) -> None:
        selected = select_examples(
            self.manifest, split="train", offset=1, limit=1, source_ids=()
        )
        self.assertEqual([row["source_id"] for row in selected], ["b"])

    def test_safe_name_removes_path_separators(self) -> None:
        self.assertEqual(safe_name("anime/00 cooking"), "anime_00_cooking")


if __name__ == "__main__":
    unittest.main()
