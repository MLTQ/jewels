"""Fit a style-stratified support-correct teacher subset for encoder audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from sol.perceptual_eval import lpips_metric, score_arms
from sol.render_streaming_continuation import frame_points
from sol.support_correct_scaling import field_structure
from stprim.data.video_io import load_video
from stprim.fit.fitter import FitConfig, fit_volume
from stprim.models.render import render_points


def _panel(frame: torch.Tensor, label: str) -> Image.Image:
    image = Image.fromarray((frame.clamp(0, 1) * 255).round().byte().numpy())
    canvas = Image.new("RGB", (image.width, image.height + 24), "black")
    canvas.paste(image, (0, 24)); ImageDraw.Draw(canvas).text((5, 6), label, fill="white")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--num-init", type=int, default=10000)
    parser.add_argument("--max-primitives", type=int, default=72000)
    parser.add_argument("--voxels", type=int, default=8192)
    parser.add_argument("--support-capacity", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device, output = torch.device(args.device), Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    selected = {}
    for item in manifest["examples"]:
        if item["split"] == "validation":
            selected.setdefault(item["style"], item)
    perceptual = lpips_metric(device)
    records, image_rows = [], []
    for style, item in sorted(selected.items()):
        video = load_video(item["video"], max_frames=item["frames"],
                           resize=(args.height, args.width), device="cpu").to(device)
        config = FitConfig(
            num_init=args.num_init, max_primitives=args.max_primitives,
            steps=args.steps, voxels_per_step=args.voxels,
            cull_mode="support_tiled", support_sigma=5.0,
            support_capacity=args.support_capacity, support_point_chunk=8192,
            support_base_resolution=32, support_level_scale=1.55,
            p1_color=True, seed=args.seed, adapt_every=100, densify_frac=0.15,
            log_every=max(1, args.steps // 10),
        )
        field, info = fit_volume(video, config, device=device, verbose=False)
        checkpoint = output / f"{style}_teacher.pt"
        torch.save({"state": field.state_dict(), "cfg": vars(config), "info": info,
                    "source": item}, checkpoint)
        indices = torch.linspace(0, len(video) - 1, args.frames).long()
        points = frame_points(len(video), indices, args.height, args.width, device=device)
        with torch.no_grad():
            rendered = render_points(
                field, points, cull_mode="support_tiled", support_sigma=5.0,
                support_capacity=args.support_capacity, support_point_chunk=8192,
                support_base_resolution=32, support_level_scale=1.55,
                background=torch.as_tensor(info["background"], device=device),
            ).reshape(args.frames, args.height, args.width, 3).cpu()
        target = video[indices].cpu()
        scored = score_arms(target, {"teacher": rendered}, perceptual)["teacher"]
        records.append({"style": style, "source_id": item["source_id"],
                        "checkpoint": str(checkpoint), "fit_seconds": info["seconds"],
                        "n_final": info["n_final"], "fit_history": info["history"],
                        "structure": field_structure(field, frames=len(video), t_scale=1.0),
                        "perceptual": scored})
        panels = [_panel(target[len(target) // 2], f"{style}: target"),
                  _panel(rendered[len(rendered) // 2], f"{style}: fitted teacher")]
        row = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height))
        row.paste(panels[0], (0, 0)); row.paste(panels[1], (panels[0].width, 0))
        image_rows.append(row)
        print("fitted teacher", style, info["n_final"], info["seconds"], flush=True)
        del field, video
        torch.cuda.empty_cache()
    macro = {
        "psnr": sum(row["perceptual"]["render_signature"]["psnr"] for row in records) / len(records),
        "ssim": sum(row["perceptual"]["render_signature"]["ssim"] for row in records) / len(records),
        "lpips": sum(row["perceptual"]["lpips_mean"] for row in records) / len(records),
        "anisotropy_median": sum(row["structure"]["anisotropy_median"] for row in records) / len(records),
        "mixed_spacetime_tilt_median": sum(row["structure"]["mixed_spacetime_tilt_median"] for row in records) / len(records),
    }
    (output / "report.json").write_text(json.dumps({
        "schema": "support-correct-encoder-teachers-v1",
        "protocol": {"styles": sorted(selected), "steps": args.steps,
                     "renderer": "support_tiled", "support_sigma": 5.0,
                     "seed": args.seed},
        "macro": macro, "records": records}, indent=2) + "\n")
    sheet = Image.new("RGB", (image_rows[0].width, sum(row.height for row in image_rows)))
    offset = 0
    for row in image_rows:
        sheet.paste(row, (0, offset)); offset += row.height
    sheet.save(output / "qualitative.png")
    print(json.dumps(macro, indent=2))


if __name__ == "__main__":
    main()
