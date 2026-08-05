"""Tests for validated prompt embedding sidecars."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from sol.prompt_embeddings import (
    build_prompt_cache,
    collect_prompts,
    load_prompt_cache,
    manifest_digest,
    save_prompt_cache,
)
from sol.ucf_prompt_manifest import VideoCandidate, build_manifest


def _manifest() -> dict:
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


class PromptEmbeddingTests(unittest.TestCase):
    def test_collects_unique_prompts_and_stable_digest(self) -> None:
        manifest = _manifest()
        prompts = collect_prompts(manifest)
        self.assertEqual(len(prompts), 4)
        self.assertEqual(manifest_digest(manifest), manifest_digest(dict(manifest)))

    def test_cache_round_trip_preserves_ownership(self) -> None:
        manifest = _manifest()
        prompts = collect_prompts(manifest)
        embeddings = torch.randn(len(prompts), 7)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        cache = build_prompt_cache(manifest, prompts, embeddings)
        self.assertEqual(len(cache.example_prompt_indices), 4)
        self.assertEqual(cache.example_prompt_indices[-1]["split"], "validation")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.pt"
            save_prompt_cache(cache, path)
            loaded = load_prompt_cache(path)
        self.assertEqual(loaded.prompts, prompts)
        self.assertTrue(torch.equal(loaded.embeddings, embeddings.float()))
        self.assertEqual(loaded.manifest_sha256, manifest_digest(manifest))

    def test_rejects_non_unit_embeddings_and_changed_prompt_order(self) -> None:
        manifest = _manifest()
        prompts = collect_prompts(manifest)
        with self.assertRaises(ValueError):
            build_prompt_cache(manifest, prompts, torch.ones(len(prompts), 3))
        embeddings = torch.eye(len(prompts))
        with self.assertRaises(ValueError):
            build_prompt_cache(manifest, tuple(reversed(prompts)), embeddings)


if __name__ == "__main__":
    unittest.main()
