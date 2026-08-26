"""Smoke tests for the learned trajectory-speaker audit."""

from __future__ import annotations

import unittest
from pathlib import Path

from sol import audit_learned_trajectory_speaker


class LearnedTrajectorySpeakerAuditTests(unittest.TestCase):
    def test_entry_point_and_protocol_exist(self) -> None:
        self.assertTrue(callable(audit_learned_trajectory_speaker.main))
        protocol = Path(__file__).parents[1] / "results" / "jewel_casting_language_v0" / (
            "PROTOCOL_LEARNED_TRAJECTORY_SPEAKER_V1.md"
        )
        self.assertTrue(protocol.exists())


if __name__ == "__main__":
    unittest.main()
