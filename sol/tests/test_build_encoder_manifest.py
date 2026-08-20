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



class SubsampleTrainTests(unittest.TestCase):
    def test_limit_is_style_and_class_balanced(self) -> None:
        from sol.build_encoder_manifest import subsample_train

        examples = [
            {"split": "train", "style": style, "class_id": class_id,
             "source_id": f"{style}_{class_id}_{index}"}
            for style in ("photoreal", "anime")
            for class_id in (0, 1)
            for index in range(5)
        ] + [{"split": "validation", "style": "photoreal", "class_id": 0,
              "source_id": "val"}]
        kept = subsample_train(examples, 8)
        train = [i for i in kept if i["split"] == "train"]
        self.assertEqual(len(train), 8)
        groups = {(i["style"], i["class_id"]) for i in train}
        self.assertEqual(len(groups), 4)
        self.assertEqual(sum(1 for i in kept if i["split"] == "validation"), 1)

    def test_limit_above_corpus_keeps_everything(self) -> None:
        from sol.build_encoder_manifest import subsample_train

        examples = [
            {"split": "train", "style": "photoreal", "class_id": 0, "source_id": "a"},
            {"split": "validation", "style": "photoreal", "class_id": 0, "source_id": "v"},
        ]
        self.assertEqual(len(subsample_train(examples, 99)), 2)

    def test_small_nested_prefix_spans_classes_and_rotates_styles(self) -> None:
        from sol.build_encoder_manifest import subsample_train

        examples = [
            {
                "split": "train",
                "style": style,
                "class_id": class_id,
                "source_id": f"{style}_{class_id}_{index}",
            }
            for style in ("anime", "cartoon", "clay", "photoreal", "render3d")
            for class_id in range(12)
            for index in range(2)
        ] + [
            {"split": "validation", "style": "anime", "class_id": 0,
             "source_id": "val"}
        ]
        small = [
            item for item in subsample_train(examples, 12)
            if item["split"] == "train"
        ]
        medium = [
            item for item in subsample_train(examples, 60)
            if item["split"] == "train"
        ]
        self.assertEqual({item["class_id"] for item in small}, set(range(12)))
        self.assertEqual({item["style"] for item in small}, {
            "anime", "cartoon", "clay", "photoreal", "render3d"
        })
        self.assertEqual(len({(item["style"], item["class_id"])
                              for item in medium}), 60)
        self.assertEqual(
            [item["source_id"] for item in medium[:12]],
            [item["source_id"] for item in small],
        )

if __name__ == "__main__":
    unittest.main()
