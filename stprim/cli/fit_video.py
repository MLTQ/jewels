"""Fit one clip and save a checkpoint.

    python cli/fit_video.py --video clip.mp4 --frames 64 --size 160
    python cli/fit_video.py --synthetic                    # falsification test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.video_io import load_video, synthetic_tube  # noqa: E402
from fit.fitter import FitConfig, fit_volume  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--size", type=int, default=256,
                    help="short side; aspect preserved")
    ap.add_argument("--num-init", type=int, default=2000)
    ap.add_argument("--max-primitives", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=3000)
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
    ap.add_argument("--no-p1", action="store_true",
                    help="constant color per primitive (P0) instead of linear ramp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="runs")
    args = ap.parse_args()

    if args.synthetic:
        vid = synthetic_tube(T=args.frames, H=args.size, W=args.size)
        name = "synthetic"
    elif args.video:
        vid = load_video(args.video, max_frames=args.frames, resize=args.size)
        name = Path(args.video).stem
    else:
        ap.error("pass --video PATH or --synthetic")

    print(f"volume: {tuple(vid.shape)}  ({vid.numel() // 3:,} voxels)")

    outdir = Path(args.out) / name
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = FitConfig(
        num_init=args.num_init,
        max_primitives=args.max_primitives,
        steps=args.steps,
        voxels_per_step=args.voxels,
        knn=args.knn,
        cull_mode=args.cull_mode,
        support_sigma=args.support_sigma,
        support_capacity=args.support_capacity,
        support_point_chunk=args.support_point_chunk,
        support_base_resolution=args.support_base_resolution,
        support_level_scale=args.support_level_scale,
        geometry_constraint=args.geometry_constraint,
        p1_color=not args.no_p1,
        t_scale=args.t_scale,
        seed=args.seed,
    )
    field, info = fit_volume(vid, cfg, device=args.device)

    torch.save(
        {"state": field.state_dict(), "cfg": vars(cfg), "info": info},
        outdir / f"fit_seed{args.seed}.pt",
    )
    (outdir / "summary.json").write_text(json.dumps(info, indent=2))
    print(
        f"final psnr~{info['history'][-1]['psnr']:.2f}  "
        f"N={info['n_final']}  {info['seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
