"""Build a self-supervised encoder manifest from generated corpus manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def subsample_train(examples: list[dict], limit: int) -> list[dict]:
    """Deterministically keep `limit` train rows, balanced over style and class.

    Round-robin across (style, class) groups so a small curve point still spans
    every domain: corpus size is the only variable the curve changes.
    """
    if limit <= 0:
        raise ValueError("train limit must be positive")
    train = [item for item in examples if item["split"] == "train"]
    held = [item for item in examples if item["split"] != "train"]
    if limit >= len(train):
        return examples
    groups: dict[tuple[str, int], list[dict]] = {}
    for item in sorted(train, key=lambda i: i["source_id"]):
        groups.setdefault((item["style"], item["class_id"]), []).append(item)
    order = sorted(groups)
    kept: list[dict] = []
    index = 0
    while len(kept) < limit:
        progressed = False
        for key in order:
            if index < len(groups[key]):
                kept.append(groups[key][index])
                progressed = True
                if len(kept) == limit:
                    break
        if not progressed:
            break
        index += 1
    return kept + held


def build_encoder_manifest(
    corpus_manifests: list[dict],
    *,
    frames: int = 49,
    style_tags: list[str] | None = None,
    train_limit: int = 0,
) -> dict:
    """Collect completed clips, holding out one evaluation clip per class/style.

    The encoder trains on render loss against the video alone, so no fitted
    field, prompt, or caption is required here: any completed clip is usable
    the moment it lands.
    """
    if frames <= 0:
        raise ValueError("frames must be positive")
    tags = style_tags or ["default"] * len(corpus_manifests)
    if len(tags) != len(corpus_manifests):
        raise ValueError("style tags must align with corpus manifests")
    examples = []
    for manifest, style in zip(corpus_manifests, tags):
        for item in manifest.get("examples", []):
            if item.get("status") != "complete":
                continue
            video = Path(item["output"])
            role = item.get("prompt_role", "train")
            examples.append(
                {
                    "class_id": int(item["class_id"]),
                    "class_name": item["class_name"],
                    "style": style,
                    "source_id": f"{style}__{item['stem']}",
                    "split": "validation" if role == "evaluation" else "train",
                    "video": str(video),
                    "frames": frames,
                    "start_frame": 0,
                    "source_prompt": item.get("source_prompt", ""),
                }
            )
    if not examples:
        raise ValueError("no completed clips found in the supplied manifests")
    if not any(item["split"] == "validation" for item in examples):
        raise ValueError("no evaluation clips completed yet; cannot hold out")
    if train_limit:
        examples = subsample_train(examples, train_limit)
    return {
        "schema": "jewel-encoder-corpus-v1",
        "frames": frames,
        "styles": sorted({item["style"] for item in examples}),
        "train_limit": train_limit or None,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", required=True,
                        help="style=path/to/corpus/manifest.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="keep only N train rows (style/class balanced) for a curve point",
    )
    args = parser.parse_args()
    manifests, tags = [], []
    for entry in args.corpus:
        style, _, path = entry.partition("=")
        if not style or not path:
            raise ValueError("--corpus entries must be style=path")
        manifests.append(json.loads(Path(path).read_text()))
        tags.append(style)
    manifest = build_encoder_manifest(
        manifests, frames=args.frames, style_tags=tags, train_limit=args.train_limit
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(manifest, indent=1))
    counts: dict[str, int] = {}
    for item in manifest["examples"]:
        counts[item["style"]] = counts.get(item["style"], 0) + 1
    print(json.dumps({
        "out": args.out,
        "total": len(manifest["examples"]),
        "train": sum(1 for i in manifest["examples"] if i["split"] == "train"),
        "validation": sum(1 for i in manifest["examples"] if i["split"] == "validation"),
        "per_style": counts,
    }))


if __name__ == "__main__":
    main()
