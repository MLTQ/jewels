"""Build a leakage-safe UCF-train/LTX-validation realizer evaluation manifest."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from sol.prompt_embeddings import (
    build_prompt_cache,
    load_prompt_cache,
    manifest_digest,
    save_prompt_cache,
)
from sol.ucf_prompt_manifest import SCHEMA as PROMPT_MANIFEST_SCHEMA


LTX_CORPUS_SCHEMA = "jewels-ltx-corpus-v1"


def build_ltx_realizer_manifest(ucf_manifest: dict, ltx_manifest: dict) -> dict:
    """Replace UCF validation rows with matched completed LTX evaluation clips."""
    if ucf_manifest.get("schema") != PROMPT_MANIFEST_SCHEMA:
        raise ValueError("unsupported UCF prompt manifest schema")
    if ltx_manifest.get("schema") != LTX_CORPUS_SCHEMA:
        raise ValueError("unsupported LTX corpus manifest schema")
    if ltx_manifest.get("source_manifest_sha256") != manifest_digest(ucf_manifest):
        raise ValueError("LTX corpus was not generated from this UCF prompt manifest")
    original_examples = ucf_manifest.get("examples", [])
    training = [item for item in original_examples if item.get("split") == "train"]
    validation = [
        item for item in original_examples if item.get("split") == "validation"
    ]
    if not training or not validation:
        raise ValueError("source manifest requires non-empty train and validation splits")
    validation_by_class = {item["class_name"]: item for item in validation}
    if len(validation_by_class) != len(validation):
        raise ValueError("source manifest requires one validation row per class")

    generated = {}
    for item in ltx_manifest.get("examples", []):
        if item.get("prompt_role") != "evaluation":
            continue
        if item.get("status") != "complete" or not item.get("output"):
            raise ValueError("every LTX evaluation scaffold must be complete")
        class_name = item.get("class_name")
        if class_name in generated:
            raise ValueError(f"multiple LTX evaluation clips for {class_name!r}")
        generated[class_name] = item
    if set(generated) != set(validation_by_class):
        raise ValueError("LTX and UCF validation classes do not match")

    replacement_by_source = {}
    frames = int(ltx_manifest.get("runtime", {}).get("num_frames", 0))
    if frames <= 0:
        raise ValueError("LTX manifest does not declare a positive frame count")
    for class_name, original in validation_by_class.items():
        scaffold = generated[class_name]
        evaluation_prompts = list(original["evaluation_prompts"])
        if scaffold.get("source_prompt") not in evaluation_prompts:
            raise ValueError(f"LTX evaluation prompt disagrees for {class_name!r}")
        replacement = deepcopy(original)
        replacement.update(
            {
                "source_group": "ltx-evaluation",
                "source_id": scaffold["stem"],
                "clip_id": int(scaffold.get("prompt_index", 0)),
                "video": scaffold["output"],
                "fit_video": scaffold["output"],
                "frame_count": frames,
                "start_frame": 0,
                "frames": frames,
                "scaffold": {
                    "schema": LTX_CORPUS_SCHEMA,
                    "prompt_role": scaffold["prompt_role"],
                    "prompt_index": scaffold["prompt_index"],
                    "generation_prompt": scaffold["generation_prompt"],
                    "seed": scaffold["seed"],
                    "receipt": scaffold["receipt"],
                    "ltx_revision": scaffold.get("result", {}).get("ltx_revision"),
                },
            }
        )
        replacement_by_source[original["source_id"]] = replacement

    derived = deepcopy(ucf_manifest)
    derived["validation_group"] = "ltx-evaluation"
    derived["validation_frames"] = frames
    derived["evaluation_domain"] = "LTX-2.3"
    derived["validation_fit_contract"] = {
        "size": 160,
        "num_init": 9000,
        "max_primitives": 72000,
        "steps": 9000,
        "voxels": 65536,
        "split_mode": "spatial",
        "recovery_every": 100,
        "windows_per_video": 1,
    }
    derived["ltx_source_manifest_sha256"] = ltx_manifest.get(
        "source_manifest_sha256"
    )
    derived["examples"] = [
        (
            replacement_by_source[item["source_id"]]
            if item.get("split") == "validation"
            else deepcopy(item)
        )
        for item in original_examples
    ]
    return derived


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucf-manifest", required=True)
    parser.add_argument("--ltx-manifest", required=True)
    parser.add_argument("--source-prompt-cache", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-prompt-cache", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ucf_manifest = json.loads(Path(args.ucf_manifest).read_text())
    ltx_manifest = json.loads(Path(args.ltx_manifest).read_text())
    source_cache = load_prompt_cache(args.source_prompt_cache)
    if source_cache.manifest_sha256 != manifest_digest(ucf_manifest):
        raise ValueError("source prompt cache does not match the UCF manifest")
    manifest = build_ltx_realizer_manifest(ucf_manifest, ltx_manifest)
    cache = build_prompt_cache(
        manifest, source_cache.prompts, source_cache.embeddings
    )
    if cache.encoder != source_cache.encoder:
        raise ValueError("derived manifest changes the prompt encoder contract")

    destination = Path(args.out_manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination)
    save_prompt_cache(cache, args.out_prompt_cache)
    print(
        json.dumps(
            {
                "manifest": str(destination),
                "prompt_cache": args.out_prompt_cache,
                "training_examples": sum(
                    item["split"] == "train" for item in manifest["examples"]
                ),
                "validation_examples": sum(
                    item["split"] == "validation"
                    for item in manifest["examples"]
                ),
                "validation_frames": manifest["validation_frames"],
                "validation_domain": manifest["evaluation_domain"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
