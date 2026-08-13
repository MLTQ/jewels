"""Tests for the explicit four-field LTX style-adaptation manifest."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import unittest

import torch

from sol.build_ltx_style_train import build_ltx_style_manifest
from sol.prompt_embeddings import build_prompt_cache, collect_prompts, manifest_digest
from sol.ucf_prompt_manifest import VideoCandidate, build_manifest


def _ucf_manifest() -> dict:
    selected = [
        VideoCandidate(
            Path(f"/Basketball/v_Basketball_g{group:02d}_c01.avi"),
            "Basketball",
            group,
            1,
            120,
        )
        for group in range(1, 5)
    ]
    return build_manifest(selected, ["Basketball"])


def _ltx_manifest(
    *,
    prompt: str = "a video of people playing basketball on a court",
    status: str = "complete",
) -> dict:
    return {
        "schema": "jewels-ltx-corpus-v1",
        "source_manifest_sha256": manifest_digest(_ucf_manifest()),
        "runtime": {"num_frames": 49},
        "examples": [
            {
                "class_id": 0,
                "class_name": "Basketball",
                "prompt_role": "evaluation",
                "prompt_index": 0,
                "source_prompt": prompt,
                "generation_prompt": f"{prompt}. Cel-shaded animation.",
                "seed": 42003,
                "stem": "00_basketball_evaluation_00_seed42003",
                "output": "/ltx/00_basketball_evaluation_00_seed42003.mp4",
                "receipt": "/ltx/00_basketball_evaluation_00_seed42003.json",
                "status": status,
                "result": {"ltx_revision": "abc123"},
            }
        ],
    }


class BuildLtxStyleTrainTests(unittest.TestCase):
    def test_builds_explicit_overlapping_train_and_reconstruction_aliases(self) -> None:
        derived = build_ltx_style_manifest(_ucf_manifest(), _ltx_manifest())
        self.assertEqual(len(derived["examples"]), 2)
        training, reconstruction = derived["examples"]
        self.assertEqual((training["split"], reconstruction["split"]), ("train", "validation"))
        self.assertNotEqual(training["source_id"], reconstruction["source_id"])
        self.assertEqual(training["video"], reconstruction["video"])
        self.assertEqual(training["shared_field_stem"], reconstruction["shared_field_stem"])
        self.assertEqual(reconstruction["overlaps_training_source_id"], training["source_id"])
        self.assertTrue(derived["source_overlap"])
        self.assertFalse(derived["validation_is_unseen"])
        self.assertEqual(derived["frames"], 49)

    def test_reuses_prompt_vectors_under_new_ownership(self) -> None:
        source = _ucf_manifest()
        prompts = collect_prompts(source)
        source_cache = build_prompt_cache(source, prompts, torch.eye(len(prompts)))
        derived = build_ltx_style_manifest(source, _ltx_manifest())
        cache = build_prompt_cache(
            derived, source_cache.prompts, source_cache.embeddings
        )
        self.assertTrue(torch.equal(cache.embeddings, source_cache.embeddings))
        self.assertNotEqual(cache.manifest_sha256, source_cache.manifest_sha256)
        self.assertEqual(
            [owner["split"] for owner in cache.example_prompt_indices],
            ["train", "validation"],
        )

    def test_accepts_exact_serialized_source_manifest_digest(self) -> None:
        source = _ucf_manifest()
        serialized = (json.dumps(source, indent=2) + "\n").encode()
        ltx = _ltx_manifest()
        ltx["source_manifest_sha256"] = hashlib.sha256(serialized).hexdigest()
        derived = build_ltx_style_manifest(
            source,
            ltx,
            source_manifest_file_sha256=ltx["source_manifest_sha256"],
        )
        self.assertEqual(
            derived["ucf_manifest_file_sha256"],
            ltx["source_manifest_sha256"],
        )

    def test_rejects_prompt_mismatch_and_incomplete_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt disagrees"):
            build_ltx_style_manifest(
                _ucf_manifest(), _ltx_manifest(prompt="wrong prompt")
            )
        with self.assertRaisesRegex(ValueError, "must be complete"):
            build_ltx_style_manifest(
                _ucf_manifest(), _ltx_manifest(status="failed")
            )


if __name__ == "__main__":
    unittest.main()
