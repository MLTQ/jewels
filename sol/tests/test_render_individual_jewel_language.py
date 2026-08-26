"""Tests for individual-Jewel qualitative record selection."""

from __future__ import annotations

import unittest

from sol.render_individual_jewel_language import select_records


class IndividualJewelRenderTests(unittest.TestCase):
    def test_lowest_fit_seed_is_selected_in_protocol_order(self) -> None:
        report = {
            "protocol": {"validation_sources": ["b", "a"]},
            "records": [
                {"source_id": "a", "fit_seed": 2},
                {"source_id": "b", "fit_seed": 1},
                {"source_id": "a", "fit_seed": 0},
            ],
        }
        selected = select_records(report)
        self.assertEqual(
            [(row["source_id"], row["fit_seed"]) for row in selected],
            [("b", 1), ("a", 0)],
        )

    def test_missing_source_is_rejected(self) -> None:
        report = {
            "protocol": {"validation_sources": ["missing"]},
            "records": [],
        }
        with self.assertRaisesRegex(ValueError, "missing"):
            select_records(report)


if __name__ == "__main__":
    unittest.main()
