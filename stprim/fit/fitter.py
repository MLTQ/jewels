"""Stochastic-voxel fitting loop for a spacetime primitive field.

Fits one clip. This is the per-instance optimization stage — the same stage
GSVC/VeGaS/GaussianVideo occupy. It exists here as the substrate for the
amortized/generative stage, not as the end goal, so it is optimized for
throughput and reproducibility rather than for last-dB reconstruction quality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dc_field

import torch

from core.params import PrimitiveField
from core.volume import make_grid, sample_indices
from fit.adapt import GradientTracker, adapt
from models.render import render_points


@dataclass
class FitConfig:
    num_init: int = 2000
    max_primitives: int = 10000
    steps: int = 3000
    lr: float = 0.01
    voxels_per_step: int = 65536
    knn: int = 64
    p1_color: bool = True
    t_scale: float = 1.0
    seed: int = 0
    # Adaptation
    adapt_every: int = 200
    adapt_until_frac: float = 0.7
    densify_frac: float = 0.08
    log_every: int = 200
    history: list = dc_field(default_factory=list)


def fit_volume(
    video: torch.Tensor,
    cfg: FitConfig,
    *,
    device: str | torch.device = "cuda",
    verbose: bool = True,
) -> tuple[PrimitiveField, dict]:
    """Fit `video` (T, H, W, 3) in [0,1] -> (field, info).

    Returns the fitted field and a dict of history/metrics.
    """
    device = torch.device(device)
    video = video.to(device)
    T, H, W, _ = video.shape

    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    grid = make_grid((T, H, W), t_scale=cfg.t_scale, device=device)
    target = video.reshape(-1, 3)
    n_voxels = grid.shape[0]

    field = PrimitiveField(
        cfg.num_init, p1_color=cfg.p1_color, device=device, generator=g
    ).to(device)

    background = torch.nn.Parameter(
        video.reshape(-1, 3).mean(0).clone().to(device)
    )

    def build_opt():
        groups = [
            {"params": [field.mu], "lr": cfg.lr * 0.5},
            {"params": [field.log_scale], "lr": cfg.lr * 0.5},
            {"params": [field.quat], "lr": cfg.lr * 0.5},
            {"params": [field.color], "lr": cfg.lr},
            {"params": [field.logit_w], "lr": cfg.lr},
            {"params": [background], "lr": cfg.lr},
        ]
        if field.p1_color:
            groups.append({"params": [field.color_grad], "lr": cfg.lr})
        return torch.optim.Adam(groups, lr=cfg.lr)

    opt = build_opt()
    tracker = GradientTracker(len(field), device)

    history = []
    t0 = time.time()
    adapt_until = int(cfg.steps * cfg.adapt_until_frac)

    for step in range(cfg.steps):
        idx = sample_indices(n_voxels, cfg.voxels_per_step, device=device)
        pts = grid[idx]
        gt = target[idx]

        pred = render_points(field, pts, knn=cfg.knn, background=background)

        loss = torch.nn.functional.mse_loss(pred, gt)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        tracker.update(field)
        opt.step()

        if (
            cfg.adapt_every
            and step > 0
            and step < adapt_until
            and step % cfg.adapt_every == 0
        ):
            stats = adapt(
                field,
                tracker,
                max_primitives=cfg.max_primitives,
                densify_frac=cfg.densify_frac,
            )
            opt = build_opt()
            if verbose:
                print(
                    f"  [adapt @ {step}] n={stats['n']} "
                    f"(+{stats['split']} -{stats['pruned']})"
                )

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            with torch.no_grad():
                psnr = -10.0 * torch.log10(loss.detach().clamp_min(1e-10))
            rec = {
                "step": step,
                "loss": float(loss.detach()),
                "psnr": float(psnr),
                "n": len(field),
            }
            history.append(rec)
            if verbose:
                print(
                    f"step {step:5d}  loss {rec['loss']:.5f}  "
                    f"psnr~{rec['psnr']:5.2f}  N={rec['n']}"
                )

    info = {
        "history": history,
        "seconds": time.time() - t0,
        "n_final": len(field),
        "shape": (T, H, W),
        # The learned background lives outside the field, so it must travel in
        # info (as a plain list: info is JSON-serialized downstream) or
        # reconstructions from a saved fit are wrong.
        "background": background.detach().cpu().tolist(),
    }
    return field, info
