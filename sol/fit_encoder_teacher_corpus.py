"""Fit a resumable source-owned support-correct teacher corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch

from sol.support_correct_scaling import field_structure
from stprim.data.video_io import load_video
from stprim.fit.fitter import FitConfig, fit_volume


def safe_name(source_id: str) -> str:
    """Keep source identity visible while producing a portable checkpoint name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id)


def select_examples(
    manifest: dict,
    *,
    split: str,
    offset: int,
    limit: int,
    source_ids: tuple[str, ...],
) -> list[dict]:
    """Select an ordered, explicit manifest subset without changing ownership."""
    if offset < 0:
        raise ValueError("teacher offset must be non-negative")
    examples = [item for item in manifest["examples"] if item["split"] == split]
    if source_ids:
        lookup = {item["source_id"]: item for item in examples}
        missing = [source_id for source_id in source_ids if source_id not in lookup]
        if missing:
            raise ValueError(f"requested sources are not in split {split!r}: {missing}")
        examples = [lookup[source_id] for source_id in source_ids]
    examples = examples[offset : offset + limit if limit else None]
    if not examples:
        raise ValueError("teacher selection is empty")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--num-init", type=int, default=10000)
    parser.add_argument("--max-primitives", type=int, default=72000)
    parser.add_argument("--voxels", type=int, default=8192)
    parser.add_argument("--support-capacity", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0 or args.steps <= 0:
        raise ValueError("limit and steps are outside their valid range")

    device, output = torch.device(args.device), Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    selected = select_examples(
        manifest,
        split=args.split,
        offset=args.offset,
        limit=args.limit,
        source_ids=tuple(args.source_id),
    )
    report_path = output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "schema": "support-correct-encoder-teacher-corpus-v1",
        "protocol": {
            "manifest": args.manifest,
            "split": args.split,
            "source_ids": [item["source_id"] for item in selected],
            "steps": args.steps,
            "renderer": "support_tiled",
            "support_sigma": 5.0,
            "seed": args.seed,
        },
        "records": [],
    }
    expected = [item["source_id"] for item in selected]
    if report["protocol"]["source_ids"] != expected:
        raise ValueError("existing teacher report owns a different source selection")
    completed = {row["source_id"] for row in report["records"]}

    for item in selected:
        source_id = item["source_id"]
        checkpoint = output / f"{safe_name(source_id)}.pt"
        if source_id in completed and checkpoint.exists():
            print("skip completed teacher", source_id, flush=True)
            continue
        video = load_video(
            item["video"],
            max_frames=int(item["frames"]),
            start_frame=int(item.get("start_frame", 0)),
            resize=(args.height, args.width),
            device="cpu",
        ).to(device)
        config = FitConfig(
            num_init=args.num_init,
            max_primitives=args.max_primitives,
            steps=args.steps,
            voxels_per_step=args.voxels,
            cull_mode="support_tiled",
            support_sigma=5.0,
            support_capacity=args.support_capacity,
            support_point_chunk=8192,
            support_base_resolution=32,
            support_level_scale=1.55,
            p1_color=True,
            seed=args.seed,
            adapt_every=100,
            densify_frac=0.15,
            log_every=max(1, args.steps // 10),
        )
        field, info = fit_volume(video, config, device=device, verbose=False)
        torch.save(
            {
                "state": field.state_dict(),
                "cfg": vars(config),
                "info": info,
                "source": item,
            },
            checkpoint,
        )
        report["records"] = [
            row for row in report["records"] if row["source_id"] != source_id
        ]
        report["records"].append({
            "source_id": source_id,
            "style": item.get("style"),
            "class_name": item.get("class_name"),
            "checkpoint": str(checkpoint),
            "fit_seconds": info["seconds"],
            "n_final": info["n_final"],
            "fit_history": info["history"],
            "structure": field_structure(field, frames=len(video), t_scale=1.0),
        })
        report["records"].sort(key=lambda row: expected.index(row["source_id"]))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print("fitted teacher", source_id, info["n_final"], info["seconds"], flush=True)
        del field, video
        torch.cuda.empty_cache()

    print(json.dumps({
        "selected": len(selected),
        "completed": len(report["records"]),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
