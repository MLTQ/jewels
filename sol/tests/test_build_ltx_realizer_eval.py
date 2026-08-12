"""Cross-domain LTX realizer evaluation-manifest tests."""

from __future__ import annotations

from pathlib import Path
import unittest

import torch

from sol.build_ltx_realizer_eval import build_ltx_realizer_manifest
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
    prompt: str = "a video of people playing basketball on a court",
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
                "generation_prompt": f"{prompt}. Realistic video.",
                "seed": 42003,
                "stem": "00_basketball_evaluation_00_seed42003",
                "output": "/ltx/00_basketball_evaluation_00_seed42003.mp4",
                "receipt": "/ltx/00_basketball_evaluation_00_seed42003.json",
                "status": "complete",
                "result": {"ltx_revision": "abc123"},
            }
        ],
    }


class BuildLtxRealizerEvalTests(unittest.TestCase):
    def test_preserves_training_and_replaces_validation(self) -> None:
        source = _ucf_manifest()
        derived = build_ltx_realizer_manifest(source, _ltx_manifest())
        self.assertEqual(derived["examples"][:3], source["examples"][:3])
        validation = derived["examples"][3]
        self.assertEqual(validation["split"], "validation")
        self.assertEqual(validation["frames"], 49)
        self.assertEqual(
            validation["source_id"], "00_basketball_evaluation_00_seed42003"
        )
        self.assertEqual(validation["scaffold"]["seed"], 42003)

    def test_reuses_prompt_rows_with_new_manifest_ownership(self) -> None:
        source = _ucf_manifest()
        prompts = collect_prompts(source)
        embeddings = torch.eye(len(prompts))
        original_cache = build_prompt_cache(source, prompts, embeddings)
        derived = build_ltx_realizer_manifest(source, _ltx_manifest())
        derived_cache = build_prompt_cache(
            derived, original_cache.prompts, original_cache.embeddings
        )
        self.assertTrue(
            torch.equal(derived_cache.embeddings, original_cache.embeddings)
        )
        self.assertNotEqual(
            derived_cache.manifest_sha256, original_cache.manifest_sha256
        )
        self.assertEqual(
            derived_cache.example_prompt_indices[-1]["source_id"],
            "00_basketball_evaluation_00_seed42003",
        )

    def test_rejects_prompt_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            build_ltx_realizer_manifest(_ucf_manifest(), _ltx_manifest("wrong prompt"))

    def test_rejects_unrelated_ltx_source_manifest(self) -> None:
        ltx = _ltx_manifest()
        ltx["source_manifest_sha256"] = "a" * 64
        with self.assertRaises(ValueError):
            build_ltx_realizer_manifest(_ucf_manifest(), ltx)


if __name__ == "__main__":
    unittest.main()
