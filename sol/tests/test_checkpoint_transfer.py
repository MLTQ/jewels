"""Tests for explicit same- and cross-manifest model initialization."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from sol.checkpoint_transfer import load_compatible_model_weights
from sol.token_grid import GridSpec


class CheckpointTransferTests(unittest.TestCase):
    def _checkpoint(self, root: str, manifest_sha256: str) -> tuple[Path, torch.nn.Linear]:
        source = torch.nn.Linear(2, 2)
        path = Path(root) / "model.pt"
        torch.save(
            {
                "model": source.state_dict(),
                "step": 17,
                "meta": {
                    "architecture": "test_architecture",
                    "model_args": {"width": 2},
                    "grid_shape": (2, 2, 2),
                    "slots_per_cell": 4,
                    "manifest_sha256": manifest_sha256,
                },
            },
            path,
        )
        return path, source

    def test_cross_manifest_transfer_is_explicit_and_model_only(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path, source = self._checkpoint(root, "a" * 64)
            destination = torch.nn.Linear(2, 2)
            provenance = load_compatible_model_weights(
                destination,
                path,
                map_location="cpu",
                architecture="test_architecture",
                model_args={"width": 2},
                grid_spec=GridSpec((2, 2, 2), 4),
                destination_manifest_sha256="b" * 64,
                allow_cross_manifest=True,
            )
            self.assertEqual(provenance["mode"], "cross_manifest_transfer")
            self.assertEqual(provenance["source_step"], 17)
            self.assertFalse(provenance["optimizer_restored"])
            for loaded, expected in zip(
                destination.parameters(), source.parameters(), strict=True
            ):
                self.assertTrue(torch.equal(loaded, expected))

    def test_same_manifest_guard_rejects_cross_manifest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path, _ = self._checkpoint(root, "a" * 64)
            with self.assertRaisesRegex(ValueError, "different manifest"):
                load_compatible_model_weights(
                    torch.nn.Linear(2, 2),
                    path,
                    map_location="cpu",
                    architecture="test_architecture",
                    model_args={"width": 2},
                    grid_spec=GridSpec((2, 2, 2), 4),
                    destination_manifest_sha256="b" * 64,
                    allow_cross_manifest=False,
                )

    def test_rejects_rank_capacity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path, _ = self._checkpoint(root, "a" * 64)
            with self.assertRaisesRegex(ValueError, "rank capacity"):
                load_compatible_model_weights(
                    torch.nn.Linear(2, 2),
                    path,
                    map_location="cpu",
                    architecture="test_architecture",
                    model_args={"width": 2},
                    grid_spec=GridSpec((2, 2, 2), 8),
                    destination_manifest_sha256="a" * 64,
                    allow_cross_manifest=False,
                )


if __name__ == "__main__":
    unittest.main()
