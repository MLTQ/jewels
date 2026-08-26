"""Tests for scene-posterior oracle source selection."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sol.audit_scene_posterior_oracle import select_oracle_sources


class ScenePosteriorOracleTests(unittest.TestCase):
    @patch("sol.audit_scene_posterior_oracle._metadata")
    def test_selects_first_source_per_exact_prompt(self, metadata) -> None:
        rows = [
            SimpleNamespace(source_id="anime-b", path="anime-b"),
            SimpleNamespace(source_id="anime-a", path="anime-a"),
            SimpleNamespace(source_id="cartoon-a", path="cartoon-a"),
        ]
        metadata.side_effect = lambda path: {
            "style": "anime" if path.startswith("anime") else "cartoon",
            "source_prompt": "dance" if path.startswith("anime") else "dog",
        }
        selected = select_oracle_sources(
            rows, ["anime-b", "anime-a", "cartoon-a"]
        )
        self.assertEqual(
            [row.source_id for row in selected], ["anime-a", "cartoon-a"]
        )

    def test_rejects_missing_registered_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            select_oracle_sources(
                [SimpleNamespace(source_id="present", path="present")],
                ["missing"],
            )


if __name__ == "__main__":
    unittest.main()
