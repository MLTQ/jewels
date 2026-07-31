"""CANONICALITY MEASUREMENT. The load-bearing check for the generative stage.

The generative stage requires that fitted primitive sets be *canonical*: the
same clip, fit twice from different initializations, must land in similar
configurations. If it doesn't, there is no learnable distribution over
primitive sets — you'd be training a prior over an arbitrary gauge choice, and
no amount of model capacity fixes that.

Two seeds, same clip, same hyperparameters. We measure agreement three ways:

  1. reconstruction PSNR    — both should fit equally well (sanity)
  2. Chamfer distance on centers in (u,v,t) — do the seeds land in the same
     places?
  3. marginal statistics of scale/anisotropy — even if individual primitives
     don't correspond, does the *distribution* match?

Interpretation:
  - low Chamfer  -> strongly canonical. Build the amortized encoder.
  - high Chamfer, matching marginals -> weakly canonical. A permutation-
    invariant (set-diffusion) prior can still work; an autoregressive one
    cannot.
  - high Chamfer, diverging marginals -> stop. Fix the fitter before going
    further.

Measured 2026-07-31 (real footage, full budget): ratio 0.621 with tightly
matching marginals — weakly canonical, set prior viable. A voronoi arm was
also measured (worse, even steelmanned) and removed; see PROJECT.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.video_io import load_video, synthetic_tube  # noqa: E402
from fit.fitter import FitConfig, fit_volume  # noqa: E402


def chamfer(a: torch.Tensor, b: torch.Tensor) -> float:
    """Symmetric mean nearest-neighbour distance between two point sets."""
    d = torch.cdist(a, b)
    return float(0.5 * (d.min(1).values.mean() + d.min(0).values.mean()))


def marginals(field) -> dict[str, float]:
    with torch.no_grad():
        s = field.scales()
        aniso = s.max(-1).values / s.min(-1).values.clamp_min(1e-6)
        return {
            "scale_med": float(s.median()),
            "scale_iqr": float(s.quantile(0.75) - s.quantile(0.25)),
            "aniso_med": float(aniso.median()),
            "weight_med": float(field.weights().median()),
        }


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
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if args.synthetic:
        vid = synthetic_tube(T=args.frames, H=args.size, W=args.size)
    elif args.video:
        vid = load_video(args.video, max_frames=args.frames, resize=args.size)
    else:
        ap.error("pass --video PATH or --synthetic")

    fields, infos = [], []
    for seed in args.seeds:
        cfg = FitConfig(
            steps=args.steps, num_init=args.num_init,
            max_primitives=args.max_primitives, seed=seed,
            voxels_per_step=args.voxels,
        )
        f, info = fit_volume(vid, cfg, device=args.device, verbose=False)
        fields.append(f)
        infos.append(info)
        print(f"seed {seed}: psnr~{info['history'][-1]['psnr']:.2f}  "
              f"N={info['n_final']}", flush=True)

    a, b = fields
    cd = chamfer(a.mu.detach(), b.mu.detach())

    # Baseline: Chamfer between one fit and a random point cloud of equal size.
    rnd = (torch.rand_like(b.mu) * 2 - 1)
    cd_rand = chamfer(a.mu.detach(), rnd)

    print("\n--- canonicalization ---")
    print(f"chamfer(fit0, fit1)   = {cd:.4f}")
    print(f"chamfer(fit0, random) = {cd_rand:.4f}   <- baseline")
    print(f"ratio                 = {cd / max(cd_rand, 1e-8):.3f}"
          "   (<<1 means canonical; ~1 means gauge freedom)")
    print("\nmarginals:")
    ma, mb = marginals(a), marginals(b)
    for k in ma:
        print(f"  {k:12s} {ma[k]:.4f}  vs  {mb[k]:.4f}")


if __name__ == "__main__":
    main()
