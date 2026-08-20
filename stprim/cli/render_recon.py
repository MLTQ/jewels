"""Fit one clip and render a side-by-side reconstruction.

    python cli/render_recon.py --synthetic --frames 24 --size 128
    python cli/render_recon.py --video clip.mp4

Metrics answer "how well"; this answers "what does it actually look like".
Writes to --out:

    compare.gif         [GT | recon] per frame, looped
    contact_sheet.png   sampled frames x {GT, recon, err x5}
    fit_seed{s}.pt      checkpoint (state + cfg + info, background included)

Only needs PIL beyond the core requirements.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from core.volume import make_grid  # noqa: E402
from data.video_io import load_video, synthetic_tube  # noqa: E402
from fit.fitter import FitConfig, fit_volume  # noqa: E402
from models.render import render_volume  # noqa: E402


def reconstruct(field, info, cfg, *, device) -> torch.Tensor:
    """Full-volume render -> (T, H, W, 3) in [0,1]."""
    T, H, W = info["shape"]
    grid = make_grid((T, H, W), t_scale=cfg.t_scale, device=device)
    background = torch.tensor(info["background"], device=device)
    with torch.no_grad():
        out = render_volume(
            field,
            grid,
            knn=cfg.knn,
            cull_mode=cfg.cull_mode,
            support_sigma=cfg.support_sigma,
            support_capacity=cfg.support_capacity,
            support_point_chunk=cfg.support_point_chunk,
            support_base_resolution=cfg.support_base_resolution,
            support_level_scale=cfg.support_level_scale,
            background=background,
        )
    return out.reshape(T, H, W, 3).clamp(0.0, 1.0)


def heat(err: torch.Tensor) -> torch.Tensor:
    """(...,) error in [0,1] -> (..., 3) black->red->yellow heatmap."""
    r = (3.0 * err).clamp(0, 1)
    g = (3.0 * err - 1.0).clamp(0, 1)
    b = (3.0 * err - 2.0).clamp(0, 1)
    return torch.stack([r, g, b], dim=-1)


def to_pil(frame: torch.Tensor, upscale: int) -> Image.Image:
    """(H, W, 3) float in [0,1] -> PIL image, nearest-upscaled."""
    arr = (frame.cpu().numpy() * 255).astype("uint8")
    img = Image.fromarray(arr)
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
    return img


def hstack(images: list[Image.Image], pad: int = 2) -> Image.Image:
    w = sum(i.width for i in images) + pad * (len(images) - 1)
    h = max(i.height for i in images)
    out = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for i in images:
        out.paste(i, (x, 0))
        x += i.width + pad
    return out


def vstack(images: list[Image.Image], pad: int = 2) -> Image.Image:
    w = max(i.width for i in images)
    h = sum(i.height for i in images) + pad * (len(images) - 1)
    out = Image.new("RGB", (w, h), (255, 255, 255))
    y = 0
    for i in images:
        out.paste(i, (0, y))
        y += i.height + pad
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--num-init", type=int, default=1500)
    ap.add_argument("--max-primitives", type=int, default=6000)
    ap.add_argument("--voxels", type=int, default=65536)
    ap.add_argument("--knn", type=int, default=64)
    ap.add_argument(
        "--cull-mode",
        choices=("knn", "support", "support_tiled", "exact"),
        default="knn",
        help="support modes are finite-support correct; exact is for tiny audits only",
    )
    ap.add_argument("--support-sigma", type=float, default=5.0)
    ap.add_argument("--support-capacity", type=int, default=512)
    ap.add_argument("--support-point-chunk", type=int, default=4096)
    ap.add_argument("--support-base-resolution", type=int, default=32)
    ap.add_argument("--support-level-scale", type=float, default=1.55)
    ap.add_argument(
        "--geometry-constraint",
        choices=("free", "axis_aligned", "isotropic"),
        default="free",
        help="causal geometry ablation; free is the production representation",
    )
    ap.add_argument("--t-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--upscale", type=int, default=2)
    ap.add_argument("--out", type=str, default="recon_out")
    args = ap.parse_args()

    if args.synthetic:
        vid = synthetic_tube(T=args.frames, H=args.size, W=args.size)
    elif args.video:
        vid = load_video(args.video, max_frames=args.frames, resize=args.size)
    else:
        ap.error("pass --video PATH or --synthetic")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    gt = vid.to(device)
    T = gt.shape[0]

    cfg = FitConfig(
        steps=args.steps, num_init=args.num_init,
        max_primitives=args.max_primitives, voxels_per_step=args.voxels,
        knn=args.knn, cull_mode=args.cull_mode,
        support_sigma=args.support_sigma,
        support_capacity=args.support_capacity,
        support_point_chunk=args.support_point_chunk,
        support_base_resolution=args.support_base_resolution,
        support_level_scale=args.support_level_scale,
        geometry_constraint=args.geometry_constraint,
        t_scale=args.t_scale, seed=args.seed,
    )
    field, info = fit_volume(gt, cfg, device=device, verbose=False)
    rec = reconstruct(field, info, cfg, device=device)
    full_psnr = -10.0 * torch.log10(
        torch.nn.functional.mse_loss(rec, gt).clamp_min(1e-10)
    )
    print(f"full-volume psnr {float(full_psnr):.2f} dB  "
          f"N={info['n_final']}  {info['seconds']:.0f}s", flush=True)
    torch.save(
        {"state": field.state_dict(), "cfg": vars(cfg), "info": info},
        outdir / f"fit_seed{args.seed}.pt",
    )

    up = args.upscale
    gif_frames = [
        hstack([to_pil(gt[t], up), to_pil(rec[t], up)]) for t in range(T)
    ]
    gif_frames[0].save(
        outdir / "compare.gif", save_all=True, append_images=gif_frames[1:],
        duration=83, loop=0,
    )

    picks = sorted({0, T // 4, T // 2, 3 * T // 4, T - 1})
    rows = []
    for row in (
        [gt[t] for t in picks],
        [rec[t] for t in picks],
        [heat((rec[t] - gt[t]).abs().mean(-1) * 5) for t in picks],
    ):
        rows.append(hstack([to_pil(f, up) for f in row]))
    vstack(rows).save(outdir / "contact_sheet.png")

    print(f"wrote {outdir}/compare.gif  (panels: GT | recon)")
    print(f"wrote {outdir}/contact_sheet.png  "
          f"(rows: GT, recon, err x5; frames {picks})")


if __name__ == "__main__":
    main()
