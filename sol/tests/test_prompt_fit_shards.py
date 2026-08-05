"""Tests for deterministic class-balanced prompt fit shards."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from sol.prompt_fit_shards import select_fit_shard, stage_fit_shard
from sol.ucf_prompt_manifest import VideoCandidate, build_manifest


def _manifest() -> dict:
    classes = ["Basketball", "HorseRiding", "PlayingGuitar", "ApplyEyeMakeup"]
    examples = [
        VideoCandidate(
            Path(f"/{name}/v_{name}_g{group:02d}_c01.avi"),
            name,
            group,
            1,
            120,
        )
        for name in classes
        for group in range(1, 5)
    ]
    return build_manifest(examples, classes)


class PromptFitShardTests(unittest.TestCase):
    def test_four_shards_are_balanced_disjoint_and_complete(self) -> None:
        manifest = _manifest()
        shards = [select_fit_shard(manifest, index, 4) for index in range(4)]
        for index, shard in enumerate(shards):
            self.assertEqual(len(shard), 4)
            self.assertEqual(len({item["class_name"] for item in shard}), 4)
            self.assertEqual({item["source_group"] for item in shard}, {index + 1})
        source_ids = [item["source_id"] for shard in shards for item in shard]
        self.assertEqual(len(source_ids), len(set(source_ids)))
        self.assertEqual(set(source_ids), {item["source_id"] for item in manifest["examples"]})

    def test_rejects_invalid_or_empty_shard(self) -> None:
        manifest = _manifest()
        with self.assertRaises(ValueError):
            select_fit_shard(manifest, 4, 4)
        with self.assertRaises(ValueError):
            select_fit_shard(manifest, 16, 17)

    def test_staging_preserves_example_and_uses_original_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Basketball" / "v_Basketball_g01_c01.avi"
            source.parent.mkdir()
            source.touch()
            example = {
                "source_id": "Basketball_g01",
                "video": str(source),
                "fit_video": "/old/stage.avi",
            }
            staged = stage_fit_shard([example], root / "stage")
            self.assertEqual(staged[0]["source_id"], example["source_id"])
            self.assertEqual(Path(staged[0]["fit_video"]).resolve(), source.resolve())


if __name__ == "__main__":
    unittest.main()
