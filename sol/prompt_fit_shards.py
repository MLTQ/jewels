"""Partition a prompt manifest into deterministic class-balanced fitting shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sol.prompt_embeddings import manifest_digest
from sol.ucf_prompt_manifest import SCHEMA as MANIFEST_SCHEMA


def select_fit_shard(manifest: dict, shard_index: int, shards: int) -> list[dict]:
    """Assign source groups round-robin within every class."""
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported prompt manifest schema")
    if shards <= 0 or not 0 <= shard_index < shards:
        raise ValueError("shard index must lie inside a positive shard count")
    examples = sorted(
        manifest["examples"],
        key=lambda item: (item["class_id"], item["source_group"], item["clip_id"]),
    )
    selected = examples[shard_index::shards]
    if not selected:
        raise ValueError("requested shard is empty")
    return selected


def stage_fit_shard(examples: list[dict], stage_dir: str | Path) -> list[dict]:
    """Create safe, idempotent fitter symlinks for one manifest shard."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for example in examples:
        source = Path(example["video"]).resolve()
        destination = stage_dir / source.name
        if destination.is_symlink():
            if destination.resolve() != source:
                raise ValueError(f"staging symlink targets the wrong video: {destination}")
        elif destination.exists():
            raise FileExistsError(f"refusing to replace staged path: {destination}")
        else:
            destination.symlink_to(source)
        record = dict(example)
        record["fit_video"] = str(destination)
        staged.append(record)
    return staged


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    selected = select_fit_shard(manifest, args.shard_index, args.shards)
    staged = stage_fit_shard(selected, args.stage_dir)
    report = {
        "schema": "ucf-prompt-fit-shard-v1",
        "source_manifest_sha256": manifest_digest(manifest),
        "shards": args.shards,
        "shard_index": args.shard_index,
        "stage_dir": str(Path(args.stage_dir)),
        "examples": staged,
    }
    output = Path(args.stage_dir) / "shard_manifest.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "shard_manifest": str(output),
                "examples": len(staged),
                "classes": sorted({item["class_name"] for item in staged}),
                "source_groups": sorted({item["source_group"] for item in staged}),
                "splits": sorted({item["split"] for item in staged}),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
