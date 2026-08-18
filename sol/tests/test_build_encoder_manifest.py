"""Tests for the self-supervised encoder manifest builder."""

from __future__ import annotations

import unittest

from sol.build_encoder_manifest import build_encoder_manifest


def _corpus(prefix: str, complete: int = 3, evaluation: bool = True) -> dict:
    examples = [
        {
            "class_id": 0,
            "class_name": "Cooking",
            "prompt_role": "train",
            "stem": f"{prefix}_train_{index:02d}",
            "output": f"/ltx/{prefix}_train_{index:02d}.mp4",
            "status": "complete" if index < complete else "pending",
            "source_prompt": "a chef chopping vegetables",
        }
        for index in range(4)
    ]
    if evaluation:
        examples.append(
            {
                "class_id": 0,
                "class_name": "Cooking",
                "prompt_role": "evaluation",
                "stem": f"{prefix}_eval_00",
                "output": f"/ltx/{prefix}_eval_00.mp4",
                "status": "complete",
                "source_prompt": "a video of cooking",
            }
        )
    return {"examples": examples}


class BuildEncoderManifestTests(unittest.TestCase):
    def test_only_completed_clips_are_included(self) -> None:
        manifest = build_encoder_manifest([_corpus("photo")], style_tags=["photoreal"])
        self.assertEqual(len(manifest["examples"]), 4)
        self.assertTrue(all("pending" not in i["source_id"] for i in manifest["examples"]))

    def test_evaluation_role_becomes_validation_split(self) -> None:
        manifest = build_encoder_manifest([_corpus("photo")], style_tags=["photoreal"])
        validation = [i for i in manifest["examples"] if i["split"] == "validation"]
        self.assertEqual(len(validation), 1)
        self.assertIn("eval", validation[0]["source_id"])

    def test_styles_are_tagged_and_ids_disambiguated(self) -> None:
        manifest = build_encoder_manifest(
            [_corpus("clip"), _corpus("clip")], style_tags=["photoreal", "anime"]
        )
        self.assertEqual(manifest["styles"], ["anime", "photoreal"])
        ids = [i["source_id"] for i in manifest["examples"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_requires_a_completed_evaluation_clip(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot hold out"):
            build_encoder_manifest(
                [_corpus("photo", evaluation=False)], style_tags=["photoreal"]
            )


if __name__ == "__main__":
    unittest.main()
