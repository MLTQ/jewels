"""Causally test the value of time-distorted primitive orientation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
STPRIM_ROOT = ROOT / "stprim"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(STPRIM_ROOT) not in sys.path:
    sys.path.insert(0, str(STPRIM_ROOT))

from data.video_io import load_video, synthetic_tube  # noqa: E402
from fit.fitter import FitConfig, fit_volume  # noqa: E402
from sol.support_correct_scaling import (  # noqa: E402
    field_structure,
    psnr,
    render_field,
)


def summarize(records: list[dict]) -> dict:
    """Compare every projected arm with free geometry at each compute budget."""
    comparisons = []
    for steps in sorted({record["steps"] for record in records}):
        at_budget = {
            record["geometry_constraint"]: record
            for record in records
            if record["steps"] == steps
        }
        free = at_budget.get("free")
        if free is None:
            continue
        for constraint, control in at_budget.items():
            if constraint == "free":
                continue
            comparisons.append(
                {
                    "steps": steps,
                    "control": constraint,
                    "free_minus_control_psnr_db": (
                        free["support_eval_psnr_db"]
                        - control["support_eval_psnr_db"]
                    ),
                    "free_minus_control_primitives": (
                        free["n_final"] - control["n_final"]
                    ),
                    "free_mixed_tilt_median": free["structure"][
                        "mixed_spacetime_tilt_median"
                    ],
                    "control_mixed_tilt_median": control["structure"][
                        "mixed_spacetime_tilt_median"
                    ],
                }
            )
    largest_axis_control = next(
        (
            item
            for item in reversed(comparisons)
            if item["control"] == "axis_aligned"
        ),
        None,
    )
    return {
        "comparisons": comparisons,
        "causal_tilt_gate": {
            "matched_axis_aligned_control_present": largest_axis_control is not None,
            "free_geometry_advantage_at_least_0_5db": bool(
                largest_axis_control
                and largest_axis_control["free_minus_control_psnr_db"] >= 0.5
            ),
            "free_geometry_uses_mixed_tilt": bool(
                largest_axis_control
                and largest_axis_control["free_mixed_tilt_median"] >= 0.2
            ),
            "projection_removes_mixed_tilt": bool(
                largest_axis_control
                and largest_axis_control["control_mixed_tilt_median"] <= 1e-5
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--video")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, nargs="+", default=[900])
    parser.add_argument(
        "--constraints",
        nargs="+",
        choices=("free", "axis_aligned", "isotropic"),
        default=["free", "axis_aligned"],
    )
    parser.add_argument("--num-init", type=int, default=300)
    parser.add_argument("--max-primitives", type=int, default=1200)
    parser.add_argument("--voxels", type=int, default=8192)
    parser.add_argument("--support-capacity", type=int, default=512)
    parser.add_argument("--support-point-chunk", type=int, default=512)
    parser.add_argument("--adapt-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.synthetic:
        video = synthetic_tube(T=args.frames, H=args.size, W=args.size)
        source_name = "synthetic_tube"
    else:
        video = load_video(args.video, max_frames=args.frames, resize=args.size)
        source_name = str(Path(args.video).resolve())
    target = video.to(args.device)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for constraint in args.constraints:
        for steps in sorted(set(args.steps)):
            cfg = FitConfig(
                num_init=args.num_init,
                max_primitives=args.max_primitives,
                steps=steps,
                voxels_per_step=args.voxels,
                cull_mode="support",
                support_capacity=args.support_capacity,
                support_point_chunk=args.support_point_chunk,
                geometry_constraint=constraint,
                seed=args.seed,
                adapt_every=args.adapt_every,
                log_every=max(1, steps // 5),
            )
            print(f"running geometry={constraint} steps={steps}", flush=True)
            field, info = fit_volume(target, cfg, device=args.device, verbose=False)
            rendered = render_field(field, info, cfg, mode="support")
            record = {
                "geometry_constraint": constraint,
                "steps": steps,
                "voxel_evaluations": steps * args.voxels,
                "fit_seconds": info["seconds"],
                "n_final": info["n_final"],
                "support_eval_psnr_db": psnr(rendered, target),
                "structure": field_structure(
                    field,
                    frames=target.shape[0],
                    t_scale=cfg.t_scale,
                    support_sigma=cfg.support_sigma,
                ),
            }
            records.append(record)
            torch.save(
                {"state": field.state_dict(), "cfg": vars(cfg), "info": info},
                output_dir / f"{constraint}_steps{steps}.pt",
            )
            report = {
                "schema": "temporal-tilt-ablation-v1",
                "source": source_name,
                "shape": list(target.shape),
                "records": records,
                "summary": summarize(records),
            }
            (output_dir / "report.json").write_text(json.dumps(report, indent=2))
            print(
                f"  PSNR={record['support_eval_psnr_db']:.2f} dB "
                f"N={record['n_final']} time={record['fit_seconds']:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
