"""Tests for prompt-generated Jewel qualitative selection and validation."""

from __future__ import annotations

import unittest

import torch

from sol.audit_jewel_casting_language import FieldRecord
from sol.render_prompt_jewel_programs import select_target_records, validate_programs


def _record(source: str, seed: int) -> FieldRecord:
    return FieldRecord(
        path=f"{source}-{seed}.pt",
        source_id=source,
        style="test",
        fit_seed=seed,
        features=torch.zeros(1, 22),
        background=torch.zeros(3),
    )


class PromptProgramRenderTests(unittest.TestCase):
    def test_lowest_fit_seed_is_selected_in_requested_order(self) -> None:
        records = [_record("b", 2), _record("a", 1), _record("b", 0), _record("a", 3)]
        selected = select_target_records(records, ["a", "b"])
        self.assertEqual([(row.source_id, row.fit_seed) for row in selected], [("a", 1), ("b", 0)])

    def test_all_control_programs_are_required(self) -> None:
        centers = torch.zeros(4, 3)
        tokens = torch.zeros(4, 3, dtype=torch.long)
        programs = {
            ("source", arm): {"centers": centers, "tokens": tokens}
            for arm in ("correct", "shuffled", "null")
        }
        validate_programs(programs, ["source"])
        del programs[("source", "null")]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_programs(programs, ["source"])

    def test_program_shapes_are_checked(self) -> None:
        programs = {
            ("source", arm): {
                "centers": torch.zeros(4, 3),
                "tokens": torch.zeros(4, 2, dtype=torch.long),
            }
            for arm in ("correct", "shuffled", "null")
        }
        with self.assertRaisesRegex(ValueError, "tokens"):
            validate_programs(programs, ["source"])


if __name__ == "__main__":
    unittest.main()
