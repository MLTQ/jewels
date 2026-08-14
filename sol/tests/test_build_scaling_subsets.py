"""Tests for class-balanced scaling-subset manifests."""

from __future__ import annotations

import unittest

from sol.build_scaling_subsets import build_subset_manifest


def _manifest() -> dict:
    examples = []
    for class_id, class_name in enumerate(("A", "B")):
        for group in (1, 2, 3):
            examples.append(
                {
                    "class_name": class_name,
                    "class_id": class_id,
                    "source_group": group,
                    "source_id": f"{class_name}_g{group:02d}",
                    "split": "train",
                }
            )
        examples.append(
            {
                "class_name": class_name,
                "class_id": class_id,
                "source_group": 9,
                "source_id": f"{class_name}_eval",
                "split": "validation",
            }
        )
    return {
        "classes": [{"class_name": "A"}, {"class_name": "B"}],
        "examples": examples,
        "schema": "test",
    }


class BuildScalingSubsetsTests(unittest.TestCase):
    def test_subset_keeps_validation_and_balances_classes(self) -> None:
        subset = build_subset_manifest(_manifest(), 2)
        train = [e for e in subset["examples"] if e["split"] == "train"]
        validation = [e for e in subset["examples"] if e["split"] == "validation"]
        self.assertEqual(len(train), 4)
        self.assertEqual(len(validation), 2)
        self.assertTrue(all(e["source_group"] <= 2 for e in train))
        self.assertEqual(subset["scaling_subset"]["train_sources"], 4)

    def test_subset_rejects_class_drop(self) -> None:
        manifest = _manifest()
        manifest["examples"] = [
            example
            for example in manifest["examples"]
            if not (example["class_name"] == "B" and example["source_group"] == 1)
        ]
        with self.assertRaisesRegex(ValueError, "class-balanced|entire class"):
            build_subset_manifest(manifest, 1)


if __name__ == "__main__":
    unittest.main()
