"""Smoke tests for the trajectory-tube oracle audit."""

from __future__ import annotations

import unittest
from pathlib import Path

from sol import audit_trajectory_tube_oracle


class TrajectoryTubeOracleAuditTests(unittest.TestCase):
    def test_entry_point_and_protocol_exist(self) -> None:
        self.assertTrue(callable(audit_trajectory_tube_oracle.main))
        protocol = Path(__file__).parents[1] / "results" / "jewel_casting_language_v0" / (
            "PROTOCOL_TRAJECTORY_TUBE_ORACLE_V1.md"
        )
        self.assertTrue(protocol.exists())


if __name__ == "__main__":
    unittest.main()
