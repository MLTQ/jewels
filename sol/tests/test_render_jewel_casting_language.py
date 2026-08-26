"""Tests for preregistered qualitative-record selection."""

from __future__ import annotations

import unittest

from sol.render_jewel_casting_language import select_qualitative_records


class CastingLanguageRenderTests(unittest.TestCase):
    def test_selects_lowest_seed_per_registered_source(self) -> None:
        report = {
            "protocol": {"validation_sources": ["a", "b"]},
            "vocabularies": {
                "64": {"records": []},
                "1024": {
                    "records": [
                        {"source_id": "a", "fit_seed": 2},
                        {"source_id": "b", "fit_seed": 1},
                        {"source_id": "a", "fit_seed": 0},
                    ]
                },
            },
        }
        selected = select_qualitative_records(report)
        self.assertEqual(
            [(row["source_id"], row["fit_seed"]) for row in selected],
            [("a", 0), ("b", 1)],
        )

    def test_requires_every_registered_source(self) -> None:
        report = {
            "protocol": {"validation_sources": ["missing"]},
            "vocabularies": {"64": {"records": []}},
        }
        with self.assertRaises(ValueError):
            select_qualitative_records(report)


if __name__ == "__main__":
    unittest.main()
