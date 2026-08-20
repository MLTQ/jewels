"""Run a matched compute curve for legacy and support-correct stage-1 fitting.

The experiment deliberately separates two questions:

1. Does center-KNN change the learned/rendered field relative to a support-safe
   candidate set?
2. Once candidates are support-safe, does reconstruction improve as optimizer
   compute increases while retaining anisotropic spacetime structure?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
STPRIM_ROOT = ROOT / "stprim"
if str(STPRIM_ROOT) not in sys.path:
    sys.path.insert(0, str(STPRIM_ROOT))

from core.volume import make_grid  # noqa: E402
from data.video_io import load_video, synthetic_tube  # noqa: E402
from fit.fitter import FitConfig, fit_volume  # noqa: E402
from models.render import render_volume  # noqa: E402


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Return clamped RGB PSNR for two equal-shaped volumes."""
    mse = torch.nn.functional.mse_loss(
        prediction.clamp(0.0, 1.0), target.clamp(0.0, 1.0)
    )
    return float(-10.0 * torch.log10(mse.clamp_min(1e-10)))


def field_structure(
    field,
    *,
    frames: int,
    t_scale: float,
    support_sigma: float = 5.0,
) -> dict[str, float]:
    """Summarize whether fitted elements are spacetime tubes rather than dots."""
    with torch.no_grad():
        scale = field.scales()
        rotation = field.rotations()
        longest_axis = scale.argmax(dim=1)
        row = torch.arange(len(field), device=field.device)
        principal = rotation[row, :, longest_axis]
        anisotropy = scale.max(dim=1).values / scale.min(dim=1).values.clamp_min(
            1e-8
        )
        temporal_alignment = principal[:, 2].abs()
        spacetime_tilt = 2.0 * temporal_alignment * torch.sqrt(
            (1.0 - temporal_alignment.square()).clamp_min(0.0)
        )

        # sqrt of the t-t covariance entry. A five-sigma diameter expressed in
        # normalized time maps to this many source frames.
        temporal_std = torch.sqrt(
            ((rotation[:, 2, :] * scale) ** 2).sum(dim=1).clamp_min(1e-12)
        )
        lifespan = support_sigma * temporal_std * max(frames - 1, 1) / t_scale
        weight = field.weights()

        return {
            "anisotropy_median": float(anisotropy.median()),
            "anisotropy_p90": float(anisotropy.quantile(0.9)),
            "principal_temporal_alignment_median": float(
                temporal_alignment.median()
            ),
            "principal_temporal_alignment_p90": float(
                temporal_alignment.quantile(0.9)
            ),
            "mixed_spacetime_tilt_median": float(spacetime_tilt.median()),
            "mixed_spacetime_tilt_p90": float(spacetime_tilt.quantile(0.9)),
            "five_sigma_lifespan_frames_median": float(lifespan.median()),
            "five_sigma_lifespan_frames_p90": float(lifespan.quantile(0.9)),
            "opacity_median": float(weight.median()),
        }


def render_field(field, info: dict, cfg: FitConfig, *, mode: str) -> torch.Tensor:
    """Render a fit using either its legacy approximation or support-safe audit."""
    grid = make_grid(info["shape"], t_scale=cfg.t_scale, device=field.device)
    background = torch.tensor(info["background"], device=field.device)
    with torch.no_grad():
        rendered = render_volume(
            field,
            grid,
            chunk=cfg.support_point_chunk,
            knn=cfg.knn,
            cull_mode=mode,
            support_sigma=cfg.support_sigma,
            support_capacity=cfg.support_capacity,
            support_point_chunk=cfg.support_point_chunk,
            support_base_resolution=cfg.support_base_resolution,
            support_level_scale=cfg.support_level_scale,
            background=background,
        )
    return rendered.reshape(*info["shape"], 3)


def summarize(records: list[dict]) -> dict:
    """Build a compact compute-scaling and culling audit from run records."""
    output = {}
    for mode in sorted({record["cull_mode"] for record in records}):
        arm = sorted(
            (record for record in records if record["cull_mode"] == mode),
            key=lambda record: record["steps"],
        )
        first, last = arm[0], arm[-1]
        output[mode] = {
            "support_psnr_gain_db": (
                last["support_eval_psnr_db"] - first["support_eval_psnr_db"]
            ),
            "wall_time_ratio": last["fit_seconds"] / max(first["fit_seconds"], 1e-9),
            "largest_run_support_psnr_db": last["support_eval_psnr_db"],
            "largest_run_renderer_gap_max_abs": last["renderer_gap_max_abs"],
            "largest_run_anisotropy_median": last["structure"][
                "anisotropy_median"
            ],
            "largest_run_temporal_alignment_p90": last["structure"][
                "principal_temporal_alignment_p90"
            ],
        }
    support = output.get("support")
    output["initial_proof_gate"] = {
        "support_arm_present": support is not None,
        "positive_compute_slope": bool(
            support and support["support_psnr_gain_db"] > 0.5
        ),
        "largest_support_run_reaches_25db": bool(
            support and support["largest_run_support_psnr_db"] >= 25.0
        ),
        "learns_nontrivial_anisotropy": bool(
            support and support["largest_run_anisotropy_median"] >= 1.5
        ),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--video")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--step-budgets", type=int, nargs="+", default=[100, 300, 900])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("knn", "support", "support_tiled"),
        default=["knn", "support"],
    )
    parser.add_argument("--num-init", type=int, default=300)
    parser.add_argument("--max-primitives", type=int, default=1200)
    parser.add_argument("--voxels", type=int, default=8192)
    parser.add_argument("--knn", type=int, default=64)
    parser.add_argument("--support-sigma", type=float, default=5.0)
    parser.add_argument("--support-capacity", type=int, default=512)
    parser.add_argument("--support-point-chunk", type=int, default=512)
    parser.add_argument("--support-base-resolution", type=int, default=32)
    parser.add_argument("--support-level-scale", type=float, default=1.55)
    parser.add_argument("--adapt-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if any(steps <= 0 for steps in args.step_budgets):
        raise ValueError("step budgets must be positive")
    if args.synthetic:
        video = synthetic_tube(T=args.frames, H=args.size, W=args.size)
        source_name = "synthetic_tube"
    else:
        video = load_video(args.video, max_frames=args.frames, resize=args.size)
        source_name = str(Path(args.video).resolve())

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = video.to(args.device)
    records: list[dict] = []

    for mode in args.modes:
        for steps in sorted(set(args.step_budgets)):
            cfg = FitConfig(
                num_init=args.num_init,
                max_primitives=args.max_primitives,
                steps=steps,
                voxels_per_step=args.voxels,
                knn=args.knn,
                cull_mode=mode,
                support_sigma=args.support_sigma,
                support_capacity=args.support_capacity,
                support_point_chunk=args.support_point_chunk,
                support_base_resolution=args.support_base_resolution,
                support_level_scale=args.support_level_scale,
                seed=args.seed,
                adapt_every=args.adapt_every,
                log_every=max(1, steps // 5),
            )
            print(f"running mode={mode} steps={steps}", flush=True)
            field, info = fit_volume(target, cfg, device=args.device, verbose=False)
            training_render = render_field(field, info, cfg, mode=mode)
            support_render = render_field(field, info, cfg, mode="support")
            renderer_gap = (training_render - support_render).abs()
            record = {
                "cull_mode": mode,
                "steps": steps,
                "voxel_evaluations": steps * args.voxels,
                "fit_seconds": info["seconds"],
                "n_final": info["n_final"],
                "training_eval_psnr_db": psnr(training_render, target),
                "support_eval_psnr_db": psnr(support_render, target),
                "renderer_gap_mean_abs": float(renderer_gap.mean()),
                "renderer_gap_max_abs": float(renderer_gap.max()),
                "structure": field_structure(
                    field,
                    frames=target.shape[0],
                    t_scale=cfg.t_scale,
                    support_sigma=cfg.support_sigma,
                ),
            }
            records.append(record)
            run_path = output_dir / f"{mode}_steps{steps}.pt"
            torch.save(
                {"state": field.state_dict(), "cfg": vars(cfg), "info": info},
                run_path,
            )
            report = {
                "schema": "support-correct-scaling-v1",
                "source": source_name,
                "shape": list(target.shape),
                "records": records,
                "summary": summarize(records),
            }
            (output_dir / "report.json").write_text(json.dumps(report, indent=2))
            print(
                f"  support PSNR={record['support_eval_psnr_db']:.2f} dB "
                f"N={record['n_final']} time={record['fit_seconds']:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
