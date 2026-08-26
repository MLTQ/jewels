"""Smoke tests for the coherent-source oracle audit."""

from __future__ import annotations

import unittest
from pathlib import Path

from sol import audit_coherent_source_oracle


class CoherentSourceOracleAuditTests(unittest.TestCase):
    def test_entry_point_and_protocol_exist(self) -> None:
        self.assertTrue(callable(audit_coherent_source_oracle.main))
        protocol = Path(__file__).parents[1] / "results" / "jewel_casting_language_v0" / (
            "PROTOCOL_COHERENT_SOURCE_ORACLE_V1.md"
        )
        self.assertTrue(protocol.exists())


if __name__ == "__main__":
    unittest.main()
