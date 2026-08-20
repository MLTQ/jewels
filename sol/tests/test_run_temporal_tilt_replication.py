"""Tests for the manifest-driven temporal-tilt replication runner."""

from __future__ import annotations

import unittest
from pathlib import Path

from sol.run_temporal_tilt_replication import build_command, validate_manifest


def manifest() -> dict:
    return {
        "schema": "temporal-tilt-replication-manifest-v1",
        "protocol": {
            "frames": 16,
            "size": 80,
            "steps": [900],
            "constraints": ["free", "axis_aligned"],
            "seeds": [0, 1, 2],
            "num_init": 500,
            "max_primitives": 2000,
            "voxels": 8192,
            "support_capacity": 1024,
            "support_point_chunk": 512,
            "adapt_every": 100,
        },
        "sources": [{"id": "clip", "path": "/data/clip.avi"}],
    }


class TemporalTiltReplicationRunnerTests(unittest.TestCase):
    def test_command_materializes_protocol_and_source(self) -> None:
        config = manifest()
        validate_manifest(config)
        command = build_command(
            config,
            config["sources"][0],
            output_dir=Path("/runs"),
            device="cuda",
        )
        self.assertIn("/data/clip.avi", command)
        self.assertEqual(command[command.index("--seeds") + 1 :][:3], ["0", "1", "2"])
        self.assertEqual(command[-1], "/runs/clip")

    def test_duplicate_source_ids_fail_before_running(self) -> None:
        config = manifest()
        config["sources"].append(dict(config["sources"][0]))
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_manifest(config)


if __name__ == "__main__":
    unittest.main()
