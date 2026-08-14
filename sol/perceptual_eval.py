"""Score rollout arms with a perceptual metric where detail restoration counts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sol.audit_prompted_washout import render_signature
from sol.guide_upsample_baseline import guide_upsample_baseline
from sol.prompt_embeddings import load_prompt_cache
from sol.render import render_exact
from sol.render_streaming_continuation import frame_points
from sol.streaming_corpus import load_prompted_fields
from sol.token_grid import GridSpec
from stprim.data.video_io import load_video


def lpips_metric(device: torch.device, net: str = "alex"):
    """Build a batched LPIPS callable; imported lazily so core tests stay light."""
    import lpips  # noqa: PLC0415

    model = lpips.LPIPS(net=net, verbose=False).to(device).eval()

    @torch.no_grad()
    def metric(candidate: torch.Tensor, target: torch.Tensor) -> list[float]:
        if candidate.shape != target.shape or candidate.ndim != 4:
            raise ValueError("videos must share one (T,H,W,3) shape")
        scores = []
        for start in range(0, len(candidate), 16):
            a = candidate[start : start + 16].permute(0, 3, 1, 2).to(device)
            b = target[start : start + 16].permute(0, 3, 1, 2).to(device)
            scores.extend(
                float(value)
                for value in model(a.clamp(0, 1), b.clamp(0, 1), normalize=True)
                .flatten()
                .cpu()
            )
        return scores

    return metric


def layout_signature(
    candidate: torch.Tensor, target: torch.Tensor, factor: int = 8
) -> dict[str, float]:
    """Score macro-layout by average-pooling away texture before comparison.

    Patch-based LPIPS rewards local texture statistics and is nearly blind to
    global structure; pooling to roughly the scaffold's own scale measures the
    opposite: where things are, not how they are textured.
    """
    if candidate.shape != target.shape or candidate.ndim != 4:
        raise ValueError("videos must share one (T,H,W,3) shape")
    if factor <= 1 or min(candidate.shape[1], candidate.shape[2]) < 2 * factor:
        raise ValueError("pool factor must leave at least two cells per axis")
    pooled = tuple(
        F.avg_pool2d(video.permute(0, 3, 1, 2).float(), factor).permute(0, 2, 3, 1)
        for video in (candidate, target)
    )
    signature = asdict(render_signature(pooled[0], pooled[1]))
    return {
        "pool_factor": float(factor),
        "layout_psnr": signature["psnr"],
        "layout_ssim": signature["ssim"],
    }


def score_arms(
    target: torch.Tensor,
    arms: dict[str, torch.Tensor],
    metric,
) -> dict[str, dict]:
    """Return per-arm perceptual, layout, and reference-based signatures."""
    report = {}
    for name, video in arms.items():
        if video.shape != target.shape:
            raise ValueError(f"arm {name!r} does not match the target shape")
        frame_scores = metric(video, target)
        factor = min(8, target.shape[1] // 2, target.shape[2] // 2)
        report[name] = {
            "lpips_mean": sum(frame_scores) / len(frame_scores),
            "lpips_per_frame": frame_scores,
            "render_signature": asdict(render_signature(video, target)),
            "layout_signature": layout_signature(video, target, factor),
        }
    return report


def _render_field(
    features: torch.Tensor,
    background: torch.Tensor,
    *,
    total_frames: int,
    frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    points = frame_points(
        total_frames, torch.arange(frames), height, width, device=device
    )
    return (
        render_exact(
            features.to(device), points, background=background.to(device)
        )
        .reshape(frames, height, width, 3)
        .cpu()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt-cache", required=True)
    parser.add_argument("--checkpoint-root", action="append", required=True)
    parser.add_argument(
        "--field",
        action="append",
        default=[],
        help="label=dir with <source_id>_generated_field.pt files",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--grid", type=int, nargs=3, default=(16, 16, 8))
    parser.add_argument("--stride-frames", type=int, default=16)
    parser.add_argument("--strides", type=int, default=3)
    parser.add_argument("--lpips-net", default="alex")
    args = parser.parse_args()

    device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text())
    prompt_cache = load_prompt_cache(args.prompt_cache)
    fields = load_prompted_fields(manifest, prompt_cache, args.checkpoint_root)
    manifest_sources = {item["source_id"]: item for item in manifest["examples"]}
    fitted_checkpoints = {}
    for root in args.checkpoint_root:
        for path in Path(root).glob("*_w000000.pt"):
            fitted_checkpoints.setdefault(path.name, path)
    completed = args.strides * args.stride_frames
    spec = GridSpec(tuple(args.grid), 1)
    metric = lpips_metric(device, args.lpips_net)
    field_dirs = []
    for entry in args.field:
        label, _, path = entry.partition("=")
        if not label or not path:
            raise ValueError("--field entries must be label=directory")
        field_dirs.append((label, Path(path)))

    records = []
    validation = [field for field in fields if field.split == "validation"]
    for source in sorted(
        validation, key=lambda item: (item.class_id, item.source_id)
    ):
        item = manifest_sources[source.source_id]
        video = load_video(
            item["video"],
            max_frames=source.frames,
            start_frame=int(item.get("start_frame", 0)),
            resize=(args.height, args.width),
            device="cpu",
        )
        target = video[:completed]
        fitted = torch.load(
            fitted_checkpoints[f"{Path(item['video']).stem}_w000000.pt"],
            map_location="cpu",
            weights_only=False,
        )
        fitted_background = torch.as_tensor(
            fitted["info"]["background"], dtype=torch.float32
        )
        arms = {
            "guide upsample baseline": guide_upsample_baseline(
                video, spec, args.stride_frames, args.strides
            ),
            "fitted jewel ceiling": _render_field(
                source.features,
                fitted_background,
                total_frames=source.frames,
                frames=completed,
                height=args.height,
                width=args.width,
                device=device,
            ),
        }
        for label, directory in field_dirs:
            saved = torch.load(
                directory / f"{source.source_id}_generated_field.pt",
                map_location="cpu",
                weights_only=False,
            )
            arms[label] = _render_field(
                saved["features"],
                torch.as_tensor(saved["background"], dtype=torch.float32),
                total_frames=source.frames,
                frames=completed,
                height=args.height,
                width=args.width,
                device=device,
            )
        scored = score_arms(target, arms, metric)
        records.append(
            {
                "source_id": source.source_id,
                "class_name": source.class_name,
                "arms": scored,
            }
        )
        print(
            source.source_id,
            {name: round(row["lpips_mean"], 4) for name, row in scored.items()},
            flush=True,
        )

    arm_names = list(records[0]["arms"])
    macro = {
        name: {
            "lpips_mean": sum(
                record["arms"][name]["lpips_mean"] for record in records
            )
            / len(records),
            "psnr": sum(
                record["arms"][name]["render_signature"]["psnr"]
                for record in records
            )
            / len(records),
            "ssim": sum(
                record["arms"][name]["render_signature"]["ssim"]
                for record in records
            )
            / len(records),
            "layout_psnr": sum(
                record["arms"][name]["layout_signature"]["layout_psnr"]
                for record in records
            )
            / len(records),
            "layout_ssim": sum(
                record["arms"][name]["layout_signature"]["layout_ssim"]
                for record in records
            )
            / len(records),
        }
        for name in arm_names
    }
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "perceptual-arm-eval-v1",
        "protocol": {
            "lpips_net": args.lpips_net,
            "height": args.height,
            "width": args.width,
            "completed_frames": completed,
            "grid_shape": list(spec.shape),
            "stride_frames": args.stride_frames,
            "reference": "manifest video resized to (height,width), first strides*stride frames",
        },
        "inputs": {
            "manifest": args.manifest,
            "checkpoint_roots": args.checkpoint_root,
            "fields": {label: str(path) for label, path in field_dirs},
        },
        "macro": macro,
        "records": records,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(macro, indent=1), flush=True)


if __name__ == "__main__":
    main()
