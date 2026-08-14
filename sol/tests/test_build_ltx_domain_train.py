"""Tests for the domain-matched LTX train manifest builder."""

from __future__ import annotations

import unittest

from sol.build_ltx_domain_train import build_domain_manifest


def _ucf_manifest() -> dict:
    return {
        "schema": "ucf-prompt-streaming-v1",
        "classes": [
            {
                "class_name": "Basketball",
                "label": "playing basketball",
                "train_prompts": ["a", "b", "c"],
                "evaluation_prompts": ["z"],
            }
        ],
        "text_encoder": {"library": "open_clip"},
    }


def _ltx_manifest() -> dict:
    examples = [
        {
            "class_id": 0,
            "class_name": "Basketball",
            "prompt_role": "train",
            "prompt_index": index,
            "source_prompt": prompt,
            "seed": 42000 + index,
            "stem": f"00_basketball_train_{index:02d}",
            "output": f"/ltx/00_basketball_train_{index:02d}.mp4",
        }
        for index, prompt in enumerate(("a", "b", "c"))
    ]
    examples.append(
        {
            "class_id": 0,
            "class_name": "Basketball",
            "prompt_role": "evaluation",
            "prompt_index": 0,
            "source_prompt": "z",
            "seed": 42003,
            "stem": "00_basketball_evaluation_00",
            "output": "/ltx/00_basketball_evaluation_00.mp4",
        }
    )
    return {"examples": examples, "source_manifest_sha256": "f" * 64}


class BuildLtxDomainTrainTests(unittest.TestCase):
    def test_builds_disjoint_split_in_class_order(self) -> None:
        manifest = build_domain_manifest(
            _ucf_manifest(), _ltx_manifest(), {"steps": 9000}
        )
        splits = [row["split"] for row in manifest["examples"]]
        self.assertEqual(splits, ["train", "train", "train", "validation"])
        self.assertTrue(manifest["validation_is_unseen"])
        self.assertFalse(manifest["source_overlap"])
        self.assertEqual(manifest["frames"], 49)
        self.assertEqual(
            [row["source_group"] for row in manifest["examples"]], [1, 2, 3, 4]
        )

    def test_rejects_prompt_class_mismatch(self) -> None:
        ltx = _ltx_manifest()
        ltx["examples"][0]["source_prompt"] = "not a class prompt"
        with self.assertRaisesRegex(ValueError, "does not belong"):
            build_domain_manifest(_ucf_manifest(), ltx, {})

    def test_rejects_missing_generation(self) -> None:
        ltx = _ltx_manifest()
        del ltx["examples"][1]
        with self.assertRaisesRegex(ValueError, "missing LTX train"):
            build_domain_manifest(_ucf_manifest(), ltx, {})


if __name__ == "__main__":
    unittest.main()
