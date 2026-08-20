"""Render an existing fitted-jewel checkpoint against its exact source window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


stprim_root = Path(__file__).resolve().parent.parent / "stprim"
if str(stprim_root) not in sys.path:
    sys.path.insert(0, str(stprim_root))

from cli.render_recon import heat, hstack, reconstruct, to_pil, vstack  # noqa: E402
from core.params import PrimitiveField  # noqa: E402
from data.video_io import load_video  # noqa: E402
from fit.fitter import FitConfig  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--video",
        help="source override for experiment checkpoints without embedded provenance",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--upscale", type=int, default=2)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if args.upscale <= 0:
        raise ValueError("upscale must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source = checkpoint.get("source")
    if args.video:
        source = {"video": args.video, "start_frame": args.start_frame}
    if source is None:
        raise ValueError(
            "checkpoint does not identify its source video window; pass --video"
        )
    cfg = FitConfig(**checkpoint["cfg"])
    info = checkpoint["info"]
    frames, height, width = info["shape"]
    video = load_video(
        source["video"],
        max_frames=frames,
        start_frame=int(source.get("start_frame", 0)),
        resize=min(height, width),
    )
    if tuple(video.shape[:3]) != tuple(info["shape"]):
        raise ValueError(
            f"decoded source shape {tuple(video.shape[:3])} != fit shape {info['shape']}"
        )
    device = torch.device(args.device)
    field = PrimitiveField(
        checkpoint["state"]["mu"].shape[0],
        p1_color=cfg.p1_color,
        device="cpu",
    )
    field.load_state_dict(checkpoint["state"])
    field = field.to(device)
    target = video.to(device)
    rendered = reconstruct(field, info, cfg, device=device)
    mse = torch.nn.functional.mse_loss(rendered, target)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    gif_frames = [
        hstack([to_pil(target[index], args.upscale), to_pil(rendered[index], args.upscale)])
        for index in range(frames)
    ]
    gif_frames[0].save(
        output / "compare.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=83,
        loop=0,
    )
    picks = sorted({0, frames // 4, frames // 2, 3 * frames // 4, frames - 1})
    rows = []
    for row in (
        [target[index] for index in picks],
        [rendered[index] for index in picks],
        [heat((rendered[index] - target[index]).abs().mean(-1) * 5) for index in picks],
    ):
        rows.append(hstack([to_pil(frame, args.upscale) for frame in row]))
    vstack(rows).save(output / "contact_sheet.png")
    report = {
        "checkpoint": args.checkpoint,
        "source": source,
        "shape": info["shape"],
        "jewels": len(field),
        "knn": cfg.knn,
        "mse": float(mse),
        "psnr": float(psnr),
        "frames": picks,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
