"""Score the trivial decode of the scaffold guide the generation stack receives."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F

from sol.audit_prompted_washout import render_signature
from sol.render_scaffold_mark_rollout import _seam_report
from sol.saliency_metrics import saliency_render_signature
from sol.token_grid import GridSpec
from sol.video_guide import video_to_cell_raster
from stprim.data.video_io import load_video


def cell_raster_to_video(
    raster: torch.Tensor,
    spec: GridSpec,
    frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Trilinear-upsample a ``(cells,3)`` guide back to ``(frames,H,W,3)`` video."""
    if raster.shape != (spec.n_cells, 3):
        raise ValueError("raster must have shape (spec.n_cells, 3)")
    if min(frames, height, width) <= 0:
        raise ValueError("frames, height, and width must be positive")
    gu, gv, gt = spec.shape
    volume = raster.float().reshape(gu, gv, gt, 3).permute(3, 2, 1, 0)[None]
    video = F.interpolate(
        volume,
        size=(frames, height, width),
        mode="trilinear",
        align_corners=False,
    )
    return video[0].permute(1, 2, 3, 0)


def guide_upsample_baseline(
    video: torch.Tensor,
    spec: GridSpec,
    stride_frames: int,
    strides: int,
) -> torch.Tensor:
    """Decode per-stride guides exactly as the rollout supplies them, trivially."""
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError("video must have shape (T,H,W,3)")
    if stride_frames <= 0 or strides <= 0:
        raise ValueError("stride_frames and strides must be positive")
    if strides * stride_frames > len(video):
        raise ValueError("video does not contain the requested complete strides")
    height, width = video.shape[1], video.shape[2]
    pieces = []
    for index in range(strides):
        frontier = index * stride_frames
        guide = video_to_cell_raster(
            video[frontier : frontier + stride_frames], spec
        )
        pieces.append(
            cell_raster_to_video(guide, spec, stride_frames, height, width)
        )
    return torch.cat(pieces, dim=0)


def evaluate_source(
    video: torch.Tensor,
    spec: GridSpec,
    stride_frames: int,
    strides: int,
    background: torch.Tensor,
) -> dict:
    """Score the guide decode against the same target slice the rollout report uses."""
    completed = strides * stride_frames
    target = video[:completed]
    baseline = guide_upsample_baseline(video, spec, stride_frames, strides)
    return {
        "completed_frames": completed,
        "render_signature": asdict(render_signature(baseline, target)),
        "saliency_signature": asdict(
            saliency_render_signature(baseline, target, background=background)
        ),
        "seams": _seam_report(baseline, target, stride_frames),
    }


def _resolve_video(path: str, roots: list[Path]) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    for root in roots:
        local = root / candidate.name
        if local.exists():
            return local
    raise FileNotFoundError(f"video not found locally: {path}")


def _contact_sheet(
    rows: list[tuple[str, torch.Tensor, torch.Tensor]],
    stride_frames: int,
    upscale: int,
) -> Image.Image:
    """Stack target/baseline frame pairs at each stride boundary for every source."""
    tiles = []
    for _, target, baseline in rows:
        frames = [0, stride_frames - 1, stride_frames, len(target) - 1]
        for video in (target, baseline):
            images = [
                Image.fromarray(
                    (video[frame].clamp(0, 1) * 255).byte().numpy()
                ).resize(
                    (video.shape[2] * upscale, video.shape[1] * upscale),
                    Image.NEAREST,
                )
                for frame in frames
            ]
            band = Image.new(
                "RGB", (sum(i.width for i in images), images[0].height)
            )
            offset = 0
            for image in images:
                band.paste(image, (offset, 0))
                offset += image.width
            tiles.append(band)
    sheet = Image.new(
        "RGB", (tiles[0].width, sum(tile.height for tile in tiles))
    )
    offset = 0
    for tile in tiles:
        sheet.paste(tile, (0, offset))
        offset += tile.height
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-root", action="append", default=[])
    parser.add_argument("--rollout-summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid", type=int, nargs=3, default=(16, 16, 8))
    parser.add_argument("--stride-frames", type=int, default=16)
    parser.add_argument("--strides", type=int, default=3)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--upscale", type=int, default=4)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    summary = json.loads(Path(args.rollout_summary).read_text())
    records = {record["source_id"]: record for record in summary["records"]}
    frames_required = args.strides * args.stride_frames + 1
    spec = GridSpec(tuple(args.grid), 1)
    roots = [Path(root) for root in args.video_root]
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    sheet_rows = []
    for item in manifest["examples"]:
        record = records.get(item["source_id"])
        if record is None:
            continue
        video = load_video(
            _resolve_video(item["video"], roots),
            max_frames=frames_required,
            start_frame=int(item.get("start_frame", 0)),
            resize=(args.height, args.width),
            device="cpu",
        )
        background = torch.tensor(
            record["background"]["fitted_reference"], dtype=torch.float32
        )
        evaluation = evaluate_source(
            video, spec, args.stride_frames, args.strides, background
        )
        pipeline = {
            panel: record["render_signatures"][panel]
            for panel in record["render_signatures"]
        }
        results.append(
            {
                "source_id": item["source_id"],
                "video": item["video"],
                "baseline": evaluation,
                "pipeline_render_signatures": pipeline,
                "baseline_minus_generated_correct_psnr": (
                    evaluation["render_signature"]["psnr"]
                    - pipeline["generated correct"]["psnr"]
                ),
            }
        )
        completed = args.strides * args.stride_frames
        sheet_rows.append(
            (
                item["source_id"],
                video[:completed],
                guide_upsample_baseline(
                    video, spec, args.stride_frames, args.strides
                ),
            )
        )

    if not results:
        raise ValueError("no manifest sources matched the rollout summary")
    macro = {
        key: sum(row["baseline"]["render_signature"][key] for row in results)
        / len(results)
        for key in results[0]["baseline"]["render_signature"]
    }
    macro_saliency = {
        key: sum(row["baseline"]["saliency_signature"][key] for row in results)
        / len(results)
        for key in results[0]["baseline"]["saliency_signature"]
    }
    report = {
        "schema": "guide-upsample-baseline-v1",
        "protocol": {
            "grid_shape": list(spec.shape),
            "stride_frames": args.stride_frames,
            "strides": args.strides,
            "height": args.height,
            "width": args.width,
            "guide_source": "target video resized to (height,width), per-stride "
            "video_to_cell_raster, trilinear upsample back",
            "reference": "same resized target slice as the rollout report",
        },
        "inputs": {
            "manifest": str(args.manifest),
            "rollout_summary": str(args.rollout_summary),
        },
        "macro_baseline_render_signature": macro,
        "macro_baseline_saliency_signature": macro_saliency,
        "macro_pipeline_render_signatures": summary["macro_render_signatures"],
        "macro_pipeline_saliency_signatures": summary.get(
            "macro_saliency_signatures"
        ),
        "records": results,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=1))
    _contact_sheet(sheet_rows, args.stride_frames, args.upscale).save(
        output_dir / "contact_sheet.png"
    )
    for row in results:
        print(
            row["source_id"],
            "baseline",
            round(row["baseline"]["render_signature"]["psnr"], 3),
            "correct",
            round(
                row["pipeline_render_signatures"]["generated correct"]["psnr"], 3
            ),
        )
    print("macro baseline psnr", round(macro["psnr"], 4))


if __name__ == "__main__":
    main()
