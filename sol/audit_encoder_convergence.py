"""Audit perceptual fidelity and spacetime structure across encoder curve points."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from PIL import Image, ImageDraw
import torch

from sol.amortized_encoder import VideoToJewelEncoder, cholesky_render
from sol.perceptual_eval import lpips_metric, score_arms
from sol.render_streaming_continuation import frame_points
from sol.token_grid import GridSpec
from stprim.data.video_io import load_video


@torch.no_grad()
def structure(prediction: dict[str, torch.Tensor], sample: int = 4096) -> dict[str, float]:
    """Summarize active covariance shape without converting the entire field."""
    count = len(prediction["centers"])
    index = torch.linspace(0, count - 1, min(sample, count),
                           device=prediction["centers"].device).long()
    weight = torch.sigmoid(prediction["logit_w"][index]).double().cpu()
    keep = weight > 0.02
    cholesky = prediction["cholesky"][index][keep].double().cpu()
    covariance = torch.linalg.inv(cholesky @ cholesky.transpose(1, 2))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    scale = eigenvalues.clamp_min(1e-12).sqrt()
    principal = eigenvectors[:, :, -1]
    anisotropy = scale[:, -1] / scale[:, 0].clamp_min(1e-8)
    temporal = principal[:, 2].abs()
    mixed = 2 * temporal * torch.sqrt((1 - temporal.square()).clamp_min(0))
    return {
        "active_fraction": float(keep.double().mean()),
        "anisotropy_median": float(anisotropy.median()),
        "anisotropy_p90": float(anisotropy.quantile(0.9)),
        "mixed_spacetime_tilt_median": float(mixed.median()),
        "mixed_spacetime_tilt_p90": float(mixed.quantile(0.9)),
        "opacity_median": float(weight[keep].median()),
    }


def _load_model(checkpoint: Path, device: torch.device) -> VideoToJewelEncoder:
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    meta = saved["meta"]
    model = VideoToJewelEncoder(
        grid_spec=GridSpec(tuple(meta["grid_shape"]), 1024), **meta["model_args"]
    ).to(device)
    model.load_state_dict(saved["model"])
    return model.eval()


def _labeled(frame: torch.Tensor, label: str) -> Image.Image:
    pixels = (frame.clamp(0, 1) * 255).round().byte().cpu().numpy()
    image = Image.fromarray(pixels)
    canvas = Image.new("RGB", (image.width, image.height + 24), "black")
    canvas.paste(image, (0, 24))
    ImageDraw.Draw(canvas).text((5, 6), label, fill="white")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sizes", type=int, nargs="+", default=(12, 60, 120))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--support-capacity", type=int, default=1024)
    args = parser.parse_args()
    root, output, device = Path(args.root), Path(args.out), torch.device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / f"n{args.sizes[0]}" / "manifest.json").read_text())
    selected = {}
    for item in manifest["examples"]:
        if item["split"] == "validation":
            selected.setdefault(item["style"], item)
    metric = lpips_metric(device)
    structure_rows, perceptual_rows = [], []
    middle_frames: dict[tuple[str, str], torch.Tensor] = {}

    for size in args.sizes:
        for seed in args.seeds:
            model = _load_model(root / f"n{size}" / f"seed{seed}" / "encoder.pt", device)
            for style, item in sorted(selected.items()):
                video = load_video(item["video"], max_frames=item["frames"],
                                   resize=(args.height, args.width), device="cpu")
                with torch.no_grad():
                    prediction = model(video.to(device))
                structure_rows.append({"train_size": size, "seed": seed,
                                       "style": style, **structure(prediction)})
                if seed != args.seeds[0]:
                    continue
                indices = torch.linspace(0, len(video) - 1, args.frames).long()
                points = frame_points(len(video), indices, args.height, args.width,
                                      device=device)
                with torch.no_grad():
                    candidate = cholesky_render(
                        prediction["centers"], prediction["cholesky"],
                        prediction["colors"], prediction["color_grads"],
                        prediction["logit_w"], points, prediction["background"],
                        point_chunk=4096, cull_mode="support_tiled",
                        support_capacity=args.support_capacity,
                    ).reshape(args.frames, args.height, args.width, 3).cpu()
                target = video[indices]
                scored = score_arms(target, {"encoder": candidate}, metric)["encoder"]
                perceptual_rows.append({"train_size": size, "seed": seed,
                                        "style": style, **scored})
                middle_frames[(style, f"n{size}")] = candidate[len(candidate) // 2]
                if size == args.sizes[0]:
                    middle_frames[(style, "target")] = target[len(target) // 2]
                print("audited", size, seed, style, flush=True)

    structure_macro = {}
    perceptual_macro = {}
    for size in args.sizes:
        rows = [row for row in structure_rows if row["train_size"] == size]
        structure_macro[str(size)] = {
            key: statistics.mean(row[key] for row in rows)
            for key in ("active_fraction", "anisotropy_median", "anisotropy_p90",
                        "mixed_spacetime_tilt_median", "mixed_spacetime_tilt_p90",
                        "opacity_median")
        }
        rows = [row for row in perceptual_rows if row["train_size"] == size]
        perceptual_macro[str(size)] = {
            "lpips_mean": statistics.mean(row["lpips_mean"] for row in rows),
            "psnr": statistics.mean(row["render_signature"]["psnr"] for row in rows),
            "ssim": statistics.mean(row["render_signature"]["ssim"] for row in rows),
            "layout_psnr": statistics.mean(row["layout_signature"]["layout_psnr"] for row in rows),
            "layout_ssim": statistics.mean(row["layout_signature"]["layout_ssim"] for row in rows),
        }
    report = {"schema": "encoder-convergence-audit-v1",
              "protocol": {"styles": sorted(selected), "examples_per_style": 1,
                           "rendered_frames": args.frames, "perceptual_seed": args.seeds[0],
                           "structure_seeds": args.seeds, "support_sigma": 5.0},
              "structure_macro": structure_macro,
              "perceptual_macro": perceptual_macro,
              "structure_records": structure_rows, "perceptual_records": perceptual_rows}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")

    styles = sorted(selected)
    rows = []
    columns = ["target"] + [f"n{size}" for size in args.sizes]
    for style in styles:
        panels = [_labeled(middle_frames[(style, column)],
                           f"{style}: {column}") for column in columns]
        row = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
        offset = 0
        for panel in panels:
            row.paste(panel, (offset, 0)); offset += panel.width
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    offset = 0
    for row in rows:
        sheet.paste(row, (0, offset)); offset += row.height
    sheet.save(output / "qualitative.png")

    import matplotlib.pyplot as plt  # noqa: PLC0415
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].plot(args.sizes, [perceptual_macro[str(n)]["lpips_mean"] for n in args.sizes], marker="o")
    axes[0].set_ylabel("LPIPS (lower is better)")
    axes[1].plot(args.sizes, [perceptual_macro[str(n)]["ssim"] for n in args.sizes], marker="o")
    axes[1].set_ylabel("SSIM (higher is better)")
    axes[2].plot(args.sizes, [structure_macro[str(n)]["anisotropy_median"] for n in args.sizes], marker="o", label="anisotropy")
    axes[2].plot(args.sizes, [structure_macro[str(n)]["mixed_spacetime_tilt_median"] for n in args.sizes], marker="s", label="mixed tilt")
    axes[2].set_ylabel("Median structure metric")
    axes[2].legend()
    for axis in axes:
        axis.set_xscale("log"); axis.set_xticks(args.sizes, labels=args.sizes)
        axis.set_xlabel("Training videos"); axis.grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(output / "perceptual_structure.png", dpi=180)
    print(json.dumps({"perceptual_macro": perceptual_macro,
                      "structure_macro": structure_macro}, indent=2))


if __name__ == "__main__":
    main()
