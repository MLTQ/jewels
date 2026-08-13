"""Build an explicit same-field LTX style-adaptation manifest and prompt cache."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from sol.build_ltx_realizer_eval import LTX_CORPUS_SCHEMA
from sol.prompt_embeddings import (
    build_prompt_cache,
    collect_prompts,
    load_prompt_cache,
    manifest_digest,
    save_prompt_cache,
)
from sol.ucf_prompt_manifest import SCHEMA as PROMPT_MANIFEST_SCHEMA


def build_ltx_style_manifest(
    ucf_manifest: dict,
    ltx_manifest: dict,
    *,
    source_manifest_file_sha256: str | None = None,
) -> dict:
    """Pair each completed LTX field with train and reconstruction-audit aliases."""
    if ucf_manifest.get("schema") != PROMPT_MANIFEST_SCHEMA:
        raise ValueError("unsupported UCF prompt manifest schema")
    if ltx_manifest.get("schema") != LTX_CORPUS_SCHEMA:
        raise ValueError("unsupported LTX corpus manifest schema")
    canonical_sha256 = manifest_digest(ucf_manifest)
    accepted_source_digests = {canonical_sha256}
    if source_manifest_file_sha256 is not None:
        accepted_source_digests.add(source_manifest_file_sha256)
    if ltx_manifest.get("source_manifest_sha256") not in accepted_source_digests:
        raise ValueError("LTX corpus was not generated from this UCF prompt manifest")

    source_validation = [
        item
        for item in ucf_manifest.get("examples", [])
        if item.get("split") == "validation"
    ]
    validation_by_class = {item["class_name"]: item for item in source_validation}
    if not validation_by_class or len(validation_by_class) != len(source_validation):
        raise ValueError("source manifest requires one validation row per class")

    generated = {}
    for item in ltx_manifest.get("examples", []):
        if item.get("prompt_role") != "evaluation":
            continue
        if item.get("status") != "complete" or not item.get("output"):
            raise ValueError("every LTX style field must be complete")
        class_name = item.get("class_name")
        if class_name in generated:
            raise ValueError(f"multiple LTX style fields for {class_name!r}")
        generated[class_name] = item
    if set(generated) != set(validation_by_class):
        raise ValueError("LTX and source validation classes do not match")

    frames = int(ltx_manifest.get("runtime", {}).get("num_frames", 0))
    if frames <= 0:
        raise ValueError("LTX manifest does not declare a positive frame count")
    training = []
    reconstruction = []
    for original in source_validation:
        class_name = original["class_name"]
        scaffold = generated[class_name]
        if scaffold.get("source_prompt") not in original["evaluation_prompts"]:
            raise ValueError(f"LTX evaluation prompt disagrees for {class_name!r}")
        common = deepcopy(original)
        common.update(
            {
                "clip_id": int(scaffold.get("prompt_index", 0)),
                "video": scaffold["output"],
                "fit_video": scaffold["output"],
                "frame_count": frames,
                "start_frame": 0,
                "frames": frames,
                "shared_field_stem": scaffold["stem"],
                "scaffold": {
                    "schema": LTX_CORPUS_SCHEMA,
                    "prompt_role": scaffold["prompt_role"],
                    "prompt_index": scaffold["prompt_index"],
                    "source_prompt": scaffold["source_prompt"],
                    "generation_prompt": scaffold["generation_prompt"],
                    "seed": scaffold["seed"],
                    "receipt": scaffold["receipt"],
                    "ltx_revision": scaffold.get("result", {}).get("ltx_revision"),
                },
            }
        )
        train_alias = deepcopy(common)
        train_alias.update(
            {
                "source_group": "ltx-style-train",
                "source_id": f"{scaffold['stem']}__style_train",
                "split": "train",
            }
        )
        validation_alias = deepcopy(common)
        validation_alias.update(
            {
                "source_group": "ltx-style-reconstruction",
                "source_id": f"{scaffold['stem']}__style_reconstruction",
                "split": "validation",
                "overlaps_training_source_id": train_alias["source_id"],
            }
        )
        train_alias["overlaps_validation_source_id"] = validation_alias["source_id"]
        training.append(train_alias)
        reconstruction.append(validation_alias)

    derived = deepcopy(ucf_manifest)
    derived.update(
        {
            "frames": frames,
            "validation_group": "same-field-training-reconstruction",
            "training_domain": "LTX-2.3 cel-shaded/rotoscoped",
            "validation_protocol": "same-field-training-reconstruction",
            "validation_is_unseen": False,
            "source_overlap": True,
            "ltx_source_manifest_sha256": ltx_manifest["source_manifest_sha256"],
            "ucf_manifest_canonical_sha256": canonical_sha256,
            "ucf_manifest_file_sha256": source_manifest_file_sha256,
            "fit_contract": {
                "size": 160,
                "num_init": 9000,
                "max_primitives": 72000,
                "steps": 9000,
                "voxels": 65536,
                "split_mode": "spatial",
                "recovery_every": 100,
                "windows_per_video": 1,
            },
            "examples": training + reconstruction,
        }
    )
    return derived


def select_ltx_style_class(manifest: dict, class_name: str) -> dict:
    """Restrict a style-adaptation manifest to one explicit physical field."""
    examples = [
        deepcopy(item)
        for item in manifest.get("examples", [])
        if item.get("class_name") == class_name
    ]
    if len(examples) != 2 or {item.get("split") for item in examples} != {
        "train",
        "validation",
    }:
        raise ValueError(
            "single-field selection requires one train and one validation alias"
        )
    stems = {item.get("shared_field_stem") for item in examples}
    if len(stems) != 1 or None in stems:
        raise ValueError("single-field aliases must share one physical field stem")
    selected = deepcopy(manifest)
    selected["classes"] = [
        deepcopy(item)
        for item in manifest.get("classes", [])
        if item.get("class_name") == class_name
    ]
    if len(selected["classes"]) != 1:
        raise ValueError(f"manifest does not declare class {class_name!r} exactly once")
    selected["examples"] = examples
    selected["single_field_overfit_class"] = class_name
    selected["validation_protocol"] = "same-field-single-class-memorization"
    selected["validation_group"] = "same-field-single-class-memorization"
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucf-manifest", required=True)
    parser.add_argument("--ltx-manifest", required=True)
    parser.add_argument("--source-prompt-cache", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-prompt-cache", required=True)
    parser.add_argument("--class-name")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ucf_path = Path(args.ucf_manifest)
    ucf_manifest = json.loads(ucf_path.read_text())
    ltx_manifest = json.loads(Path(args.ltx_manifest).read_text())
    source_cache = load_prompt_cache(args.source_prompt_cache)
    if source_cache.manifest_sha256 != manifest_digest(ucf_manifest):
        raise ValueError("source prompt cache does not match the UCF manifest")
    manifest = build_ltx_style_manifest(
        ucf_manifest,
        ltx_manifest,
        source_manifest_file_sha256=hashlib.sha256(ucf_path.read_bytes()).hexdigest(),
    )
    if args.class_name:
        manifest = select_ltx_style_class(manifest, args.class_name)
    expected_prompts = collect_prompts(manifest)
    prompt_lookup = {
        prompt: index for index, prompt in enumerate(source_cache.prompts)
    }
    if set(expected_prompts) - set(prompt_lookup):
        raise ValueError("source prompt cache lacks selected manifest prompts")
    prompt_rows = [prompt_lookup[prompt] for prompt in expected_prompts]
    cache = build_prompt_cache(
        manifest,
        expected_prompts,
        source_cache.embeddings[prompt_rows],
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
                "training_examples": len(
                    [item for item in manifest["examples"] if item["split"] == "train"]
                ),
                "reconstruction_examples": len(
                    [
                        item
                        for item in manifest["examples"]
                        if item["split"] == "validation"
                    ]
                ),
                "frames": manifest["frames"],
                "validation_protocol": manifest["validation_protocol"],
                "source_overlap": manifest["source_overlap"],
                "single_field_overfit_class": manifest.get(
                    "single_field_overfit_class"
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
