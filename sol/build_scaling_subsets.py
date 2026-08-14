"""Write class-balanced train-subset manifests for data-scaling measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_subset_manifest(manifest: dict, max_source_group: int) -> dict:
    """Keep validation untouched and train sources up to one group index per class."""
    if max_source_group <= 0:
        raise ValueError("max_source_group must be positive")
    examples = []
    kept_groups: dict[str, set[int]] = {}
    for example in manifest["examples"]:
        if example["split"] == "validation":
            examples.append(example)
            continue
        if example["split"] != "train":
            raise ValueError(f"unknown split {example['split']!r}")
        if int(example["source_group"]) <= max_source_group:
            examples.append(example)
            kept_groups.setdefault(example["class_name"], set()).add(
                int(example["source_group"])
            )
    class_names = {item["class_name"] for item in manifest["classes"]}
    if set(kept_groups) != class_names:
        raise ValueError("subset dropped an entire class")
    counts = {len(groups) for groups in kept_groups.values()}
    if len(counts) != 1:
        raise ValueError("subset is not class-balanced")
    subset = dict(manifest)
    subset["examples"] = examples
    subset["scaling_subset"] = {
        "max_source_group": max_source_group,
        "train_sources": sum(
            1 for example in examples if example["split"] == "train"
        ),
        "parent_examples": len(manifest["examples"]),
    }
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-source-group", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    subset = build_subset_manifest(manifest, args.max_source_group)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(subset, indent=1))
    print(
        json.dumps(
            {
                "out": args.out,
                **subset["scaling_subset"],
                "validation_sources": sum(
                    1
                    for example in subset["examples"]
                    if example["split"] == "validation"
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
