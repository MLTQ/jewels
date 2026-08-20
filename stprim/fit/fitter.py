"""Stochastic-voxel fitting loop for a spacetime primitive field.

Fits one clip. This is the per-instance optimization stage — the same stage
GSVC/VeGaS/GaussianVideo occupy. It exists here as the substrate for the
amortized/generative stage, not as the end goal, so it is optimized for
throughput and reproducibility rather than for last-dB reconstruction quality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable

import torch

from core.params import PrimitiveField
from core.volume import make_grid, sample_indices
from fit.adapt import GradientTracker, adapt
from fit.recovery import tensors_to_cpu
from models.render import render_points


@dataclass
class FitConfig:
    num_init: int = 2000
    max_primitives: int = 10000
    steps: int = 3000
    lr: float = 0.01
    voxels_per_step: int = 65536
    knn: int = 64
    cull_mode: str = "knn"
    support_sigma: float = 5.0
    support_capacity: int = 512
    support_point_chunk: int = 4096
    support_base_resolution: int = 32
    support_level_scale: float = 1.55
    geometry_constraint: str = "free"
    p1_color: bool = True
    t_scale: float = 1.0
    seed: int = 0
    # Adaptation
    adapt_every: int = 200
    adapt_until_frac: float = 0.7
    densify_frac: float = 0.08
    split_mode: str = "isotropic"
    log_every: int = 200
    history: list = dc_field(default_factory=list)


@torch.no_grad()
def project_geometry_(field: PrimitiveField, constraint: str) -> None:
    """Project a field onto a geometry ablation manifold in place.

    ``axis_aligned`` removes spacetime tilt while retaining independent spatial
    and temporal scales. ``isotropic`` additionally forces all three scales to
    match. These are causal controls, not proposed production modes.
    """
    if constraint == "free":
        return
    if constraint not in {"axis_aligned", "isotropic"}:
        raise ValueError(f"unknown geometry_constraint {constraint!r}")
    field.quat.zero_()
    field.quat[:, 0] = 1.0
    if constraint == "isotropic":
        field.log_scale[:] = field.log_scale.mean(dim=1, keepdim=True)


def fit_volume(
    video: torch.Tensor,
    cfg: FitConfig,
    *,
    device: str | torch.device = "cuda",
    verbose: bool = True,
    resume_state: dict[str, Any] | None = None,
    checkpoint_every: int = 0,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[PrimitiveField, dict]:
    """Fit `video` (T, H, W, 3) in [0,1] -> (field, info).

    Returns the fitted field and a dict of history/metrics.
    """
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    device = torch.device(device)
    video = video.to(device)
    T, H, W, _ = video.shape

    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    grid = make_grid((T, H, W), t_scale=cfg.t_scale, device=device)
    target = video.reshape(-1, 3)
    n_voxels = grid.shape[0]

    initial_count = cfg.num_init
    if resume_state is not None:
        initial_count = int(resume_state["field_state"]["mu"].shape[0])
    field = PrimitiveField(
        initial_count, p1_color=cfg.p1_color, device=device, generator=g
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

    history: list[dict[str, Any]] = []
    start_step = 0
    elapsed_before = 0.0
    if resume_state is not None:
        if int(resume_state.get("state_version", -1)) != 1:
            raise ValueError("unsupported fit recovery state version")
        if tuple(resume_state["shape"]) != (T, H, W):
            raise ValueError(
                f"recovery shape {tuple(resume_state['shape'])} does not match "
                f"video shape {(T, H, W)}"
            )
        start_step = int(resume_state["next_step"])
        if not 0 <= start_step <= cfg.steps:
            raise ValueError(
                f"recovery next_step {start_step} is outside 0..{cfg.steps}"
            )
        field.load_state_dict(resume_state["field_state"])
        with torch.no_grad():
            background.copy_(resume_state["background"].to(device))
        opt.load_state_dict(resume_state["optimizer_state"])
        tracker.accum = resume_state["tracker_accum"].to(device).clone()
        tracker.count = int(resume_state["tracker_count"])
        if tracker.accum.shape != (len(field),):
            raise ValueError("recovery gradient tracker does not match primitive count")
        history = [dict(record) for record in resume_state["history"]]
        elapsed_before = float(resume_state["elapsed_seconds"])
        g.set_state(resume_state["generator_state"].cpu())

    project_geometry_(field, cfg.geometry_constraint)

    t0 = time.time()
    adapt_until = int(cfg.steps * cfg.adapt_until_frac)

    for step in range(start_step, cfg.steps):
        # This dedicated CPU generator is the sole source of fit randomness.
        # Its one saved state resumes identically on CUDA, MPS, or CPU.
        idx = sample_indices(
            n_voxels,
            cfg.voxels_per_step,
            device="cpu",
            generator=g,
        ).to(device)
        pts = grid[idx]
        gt = target[idx]

        pred = render_points(
            field,
            pts,
            knn=cfg.knn,
            cull_mode=cfg.cull_mode,
            support_sigma=cfg.support_sigma,
            support_capacity=cfg.support_capacity,
            support_point_chunk=cfg.support_point_chunk,
            support_base_resolution=cfg.support_base_resolution,
            support_level_scale=cfg.support_level_scale,
            background=background,
        )

        loss = torch.nn.functional.mse_loss(pred, gt)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        tracker.update(field)
        opt.step()
        project_geometry_(field, cfg.geometry_constraint)

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
                split_mode=cfg.split_mode,
                generator=g,
            )
            project_geometry_(field, cfg.geometry_constraint)
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

        next_step = step + 1
        if (
            checkpoint_every > 0
            and checkpoint_callback is not None
            and next_step % checkpoint_every == 0
        ):
            checkpoint_callback(
                {
                    "state_version": 1,
                    "next_step": next_step,
                    "shape": (T, H, W),
                    "field_state": tensors_to_cpu(field.state_dict()),
                    "background": background.detach().cpu().clone(),
                    "optimizer_state": tensors_to_cpu(opt.state_dict()),
                    "tracker_accum": tracker.accum.detach().cpu().clone(),
                    "tracker_count": tracker.count,
                    "history": [dict(record) for record in history],
                    "elapsed_seconds": elapsed_before + time.time() - t0,
                    "generator_state": g.get_state().clone(),
                }
            )

    info = {
        "history": history,
        "seconds": elapsed_before + time.time() - t0,
        "n_final": len(field),
        "shape": (T, H, W),
        # The learned background lives outside the field, so it must travel in
        # info (as a plain list: info is JSON-serialized downstream) or
        # reconstructions from a saved fit are wrong.
        "background": background.detach().cpu().tolist(),
    }
    return field, info
