"""Build the domain-matched LTX-train manifest with a physically disjoint eval split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sol.prompt_embeddings import (
    build_prompt_cache,
    load_prompt_cache,
    manifest_digest,
    save_prompt_cache,
)


def build_domain_manifest(
    ucf_manifest: dict, ltx_manifest: dict, fit_contract: dict
) -> dict:
    """Twelve LTX train generations as train rows, four eval generations as validation."""
    generations: dict[tuple[str, str, int], dict] = {}
    for item in ltx_manifest["examples"]:
        key = (item["class_name"], item["prompt_role"], int(item["prompt_index"]))
        if key in generations:
            raise ValueError(f"duplicate LTX generation for {key}")
        generations[key] = item
    examples = []
    for ucf_class in ucf_manifest["classes"]:
        class_name = ucf_class["class_name"]
        train_prompts = list(ucf_class["train_prompts"])
        evaluation_prompts = list(ucf_class["evaluation_prompts"])
        rows = []
        for index in range(len(train_prompts)):
            item = generations.get((class_name, "train", index))
            if item is None:
                raise ValueError(f"missing LTX train generation {class_name}/{index}")
            rows.append((item, "train", index + 1))
        evaluation = generations.get((class_name, "evaluation", 0))
        if evaluation is None:
            raise ValueError(f"missing LTX evaluation generation for {class_name}")
        rows.append((evaluation, "validation", len(train_prompts) + 1))
        for item, split, group in rows:
            source_prompt = " ".join(item["source_prompt"].split())
            expected = train_prompts if split == "train" else evaluation_prompts
            if source_prompt not in {" ".join(p.split()) for p in expected}:
                raise ValueError(
                    f"generation prompt does not belong to {class_name} {split}"
                )
            examples.append(
                {
                    "class_id": int(item["class_id"]),
                    "class_name": class_name,
                    "label": ucf_class["label"],
                    "source_group": group,
                    "source_id": item["stem"],
                    "clip_id": int(item["seed"]),
                    "split": split,
                    "video": item["output"],
                    "frame_count": 49,
                    "start_frame": 0,
                    "frames": 49,
                    "train_prompts": train_prompts,
                    "evaluation_prompts": evaluation_prompts,
                }
            )
    return {
        "schema": ucf_manifest["schema"],
        "frames": 49,
        "validation_group": "ltx-evaluation-prompts",
        "classes": ucf_manifest["classes"],
        "text_encoder": ucf_manifest["text_encoder"],
        "fit_contract": fit_contract,
        "examples": examples,
        "training_domain": "LTX-2.3 photoreal prompt corpus",
        "validation_is_unseen": True,
        "source_overlap": False,
        "ltx_source_manifest_sha256": ltx_manifest["source_manifest_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ucf-manifest", required=True)
    parser.add_argument("--ltx-manifest", required=True)
    parser.add_argument("--source-prompt-cache", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-prompt-cache", required=True)
    args = parser.parse_args()
    ucf_manifest = json.loads(Path(args.ucf_manifest).read_text())
    ltx_manifest = json.loads(Path(args.ltx_manifest).read_text())
    parent = load_prompt_cache(args.source_prompt_cache)
    if parent.manifest_sha256 != manifest_digest(ucf_manifest):
        raise ValueError("prompt cache does not own the UCF manifest")
    fit_contract = {
        "size": 160,
        "num_init": 9000,
        "max_primitives": 72000,
        "steps": 9000,
        "voxels": 65536,
        "split_mode": "spatial",
        "recovery_every": 100,
        "windows_per_video": 1,
    }
    manifest = build_domain_manifest(ucf_manifest, ltx_manifest, fit_contract)
    cache = build_prompt_cache(manifest, parent.prompts, parent.embeddings)
    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_manifest).write_text(json.dumps(manifest, indent=1))
    save_prompt_cache(cache, args.out_prompt_cache)
    print(
        json.dumps(
            {
                "out_manifest": args.out_manifest,
                "train_rows": sum(
                    1 for row in manifest["examples"] if row["split"] == "train"
                ),
                "validation_rows": sum(
                    1 for row in manifest["examples"] if row["split"] == "validation"
                ),
                "manifest_sha256": cache.manifest_sha256,
            }
        )
    )


if __name__ == "__main__":
    main()
