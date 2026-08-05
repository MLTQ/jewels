"""Tests for balanced UCF prompt-manifest selection and splitting."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from sol.ucf_prompt_manifest import (
    SCHEMA,
    VideoCandidate,
    build_manifest,
    parse_ucf_video,
    select_balanced_candidates,
    stage_candidates,
)


class UCFPromptManifestTests(unittest.TestCase):
    def test_parse_validates_directory_class_and_ids(self) -> None:
        candidate = parse_ucf_video(
            "/data/Basketball/v_Basketball_g02_c05.avi", 144
        )
        self.assertEqual(candidate.class_name, "Basketball")
        self.assertEqual((candidate.group, candidate.clip), (2, 5))
        with self.assertRaises(ValueError):
            parse_ucf_video("/data/HorseRiding/v_Basketball_g02_c05.avi", 144)

    def test_selection_is_longest_balanced_and_split_by_group(self) -> None:
        classes = ["Basketball", "HorseRiding"]
        candidates = []
        for class_name in classes:
            for group in range(1, 5):
                candidates.extend(
                    (
                        VideoCandidate(
                            Path(f"/{class_name}/v_{class_name}_g{group:02d}_c01.avi"),
                            class_name,
                            group,
                            1,
                            110,
                        ),
                        VideoCandidate(
                            Path(f"/{class_name}/v_{class_name}_g{group:02d}_c02.avi"),
                            class_name,
                            group,
                            2,
                            140,
                        ),
                    )
                )
        selected = select_balanced_candidates(candidates, classes)
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(item.clip == 2 for item in selected))
        manifest = build_manifest(selected, classes, validation_group=4)
        self.assertEqual(manifest["schema"], SCHEMA)
        self.assertEqual(
            sum(item["split"] == "train" for item in manifest["examples"]), 6
        )
        self.assertEqual(
            sum(item["split"] == "validation" for item in manifest["examples"]),
            2,
        )
        for item in manifest["examples"]:
            self.assertTrue(set(item["train_prompts"]).isdisjoint(item["evaluation_prompts"]))

    def test_selection_rejects_missing_or_short_source_group(self) -> None:
        candidates = [
            VideoCandidate(
                Path(f"/Basketball/v_Basketball_g{group:02d}_c01.avi"),
                "Basketball",
                group,
                1,
                95 if group == 3 else 120,
            )
            for group in range(1, 5)
        ]
        with self.assertRaises(ValueError):
            select_balanced_candidates(candidates, ["Basketball"])

    def test_staging_is_idempotent_and_rejects_wrong_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "Basketball" / "v_Basketball_g01_c01.avi"
            first.parent.mkdir()
            first.touch()
            candidate = VideoCandidate(first, "Basketball", 1, 1, 120)
            stage = root / "stage"
            staged = stage_candidates([candidate], stage)
            self.assertEqual(staged[first].resolve(), first.resolve())
            stage_candidates([candidate], stage)
            staged[first].unlink()
            staged[first].write_text("not a symlink")
            with self.assertRaises(FileExistsError):
                stage_candidates([candidate], stage)


if __name__ == "__main__":
    unittest.main()
