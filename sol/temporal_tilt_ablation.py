"""Causally test the value of time-distorted primitive orientation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
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


_T_CRITICAL_975 = (
    0.0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def mean_confidence_interval(values: list[float]) -> dict:
    """Return paired mean, sample spread, and a two-sided 95% t interval."""
    if not values:
        return {
            "n": 0,
            "mean": None,
            "sample_std": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    mean = statistics.mean(values)
    if len(values) == 1:
        return {
            "n": 1,
            "mean": mean,
            "sample_std": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    sample_std = statistics.stdev(values)
    degrees = len(values) - 1
    critical = _T_CRITICAL_975[min(degrees, 30)] if degrees <= 30 else 1.96
    radius = critical * sample_std / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "sample_std": sample_std,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
    }


def reconstruction_error(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    """Report global, tail, worst-frame, and target-motion-region RGB errors."""
    pixel_error = (prediction.clamp(0.0, 1.0) - target).abs().mean(dim=-1)
    frame_error = pixel_error.mean(dim=(1, 2))
    motion = torch.zeros_like(pixel_error)
    motion[1:] = (target[1:] - target[:-1]).abs().mean(dim=-1)
    threshold = motion.quantile(0.8)
    moving = motion >= threshold
    return {
        "rgb_mae": float(pixel_error.mean()),
        "pixel_mae_p95": float(pixel_error.quantile(0.95)),
        "pixel_mae_max": float(pixel_error.max()),
        "worst_frame_rgb_mae": float(frame_error.max()),
        "motion_top20_rgb_mae": float(pixel_error[moving].mean()),
    }


def field_storage(field, target: torch.Tensor) -> dict[str, float | int]:
    """Describe raw tensor storage and source-video coverage per primitive."""
    parameter_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in field.state_dict().values()
    ) + 3 * target.element_size()
    raw_uint8_bytes = target.numel()
    return {
        "parameter_bytes": parameter_bytes,
        "parameter_megabytes": parameter_bytes / 1_000_000.0,
        "raw_uint8_video_bytes": raw_uint8_bytes,
        "raw_uint8_to_parameter_ratio": raw_uint8_bytes / parameter_bytes,
        "video_voxels_per_primitive": target.numel() / 3 / len(field),
    }


def source_fingerprint(path: Path) -> dict[str, str | int]:
    """Bind a source video to its path, byte size, and SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def environment_report(device: torch.device) -> dict[str, str]:
    """Record the numerical runtime and selected accelerator."""
    accelerator = "cpu"
    accelerator_uuid = ""
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        accelerator = properties.name
        accelerator_uuid = str(getattr(properties, "uuid", ""))
    return {
        "torch_version": torch.__version__,
        "device": str(device),
        "accelerator": accelerator,
        "accelerator_uuid": accelerator_uuid,
    }


def summarize(records: list[dict]) -> dict:
    """Compare every projected arm with free geometry for each seed and budget."""
    comparisons = []
    keys = sorted(
        {
            (record["steps"], int(record.get("seed", 0)))
            for record in records
        }
    )
    for steps, seed in keys:
        at_budget = {
            record["geometry_constraint"]: record
            for record in records
            if record["steps"] == steps and int(record.get("seed", 0)) == seed
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
                    "seed": seed,
                    "control": constraint,
                    "free_support_psnr_db": free["support_eval_psnr_db"],
                    "control_support_psnr_db": control["support_eval_psnr_db"],
                    "free_minus_control_psnr_db": (
                        free["support_eval_psnr_db"]
                        - control["support_eval_psnr_db"]
                    ),
                    "free_primitives": free["n_final"],
                    "control_primitives": control["n_final"],
                    "free_minus_control_primitives": (
                        free["n_final"] - control["n_final"]
                    ),
                    "free_parameter_bytes": free.get("storage", {}).get(
                        "parameter_bytes"
                    ),
                    "control_parameter_bytes": control.get("storage", {}).get(
                        "parameter_bytes"
                    ),
                    "free_psnr_db_per_1000_primitives": free.get(
                        "psnr_db_per_1000_primitives"
                    ),
                    "control_psnr_db_per_1000_primitives": control.get(
                        "psnr_db_per_1000_primitives"
                    ),
                    "free_psnr_db_per_parameter_megabyte": free.get(
                        "psnr_db_per_parameter_megabyte"
                    ),
                    "control_psnr_db_per_parameter_megabyte": control.get(
                        "psnr_db_per_parameter_megabyte"
                    ),
                    "free_mixed_tilt_median": free["structure"][
                        "mixed_spacetime_tilt_median"
                    ],
                    "control_mixed_tilt_median": control["structure"][
                        "mixed_spacetime_tilt_median"
                    ],
                    "control_minus_free_rgb_mae": (
                        control.get("error", {}).get("rgb_mae", 0.0)
                        - free.get("error", {}).get("rgb_mae", 0.0)
                    ),
                    "control_minus_free_motion_top20_rgb_mae": (
                        control.get("error", {}).get("motion_top20_rgb_mae", 0.0)
                        - free.get("error", {}).get("motion_top20_rgb_mae", 0.0)
                    ),
                }
            )
    largest_steps = max((item["steps"] for item in comparisons), default=None)
    largest_axis = [
        item
        for item in comparisons
        if item["control"] == "axis_aligned" and item["steps"] == largest_steps
    ]
    delta_stats = mean_confidence_interval(
        [item["free_minus_control_psnr_db"] for item in largest_axis]
    )
    free_wins = sum(item["free_minus_control_psnr_db"] > 0 for item in largest_axis)
    mean_free_tilt = (
        statistics.mean(item["free_mixed_tilt_median"] for item in largest_axis)
        if largest_axis
        else 0.0
    )
    return {
        "comparisons": comparisons,
        "largest_budget_axis_aligned": {
            "steps": largest_steps,
            "paired_psnr_delta_db": delta_stats,
            "free_wins": free_wins,
            "pair_count": len(largest_axis),
        },
        "causal_tilt_gate": {
            "matched_axis_aligned_control_present": bool(largest_axis),
            "free_geometry_advantage_at_least_0_5db": bool(
                delta_stats["mean"] is not None and delta_stats["mean"] >= 0.5
            ),
            "free_geometry_uses_mixed_tilt": bool(
                largest_axis and mean_free_tilt >= 0.2
            ),
            "projection_removes_mixed_tilt": bool(
                largest_axis
                and max(item["control_mixed_tilt_median"] for item in largest_axis)
                <= 1e-5
            ),
        },
    }


def write_report(path: Path, report: dict) -> None:
    """Atomically replace an incremental JSON report."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--video")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--start-frame", type=int, default=0)
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
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.synthetic:
        video = synthetic_tube(T=args.frames, H=args.size, W=args.size)
        source_name = "synthetic_tube"
        fingerprint = {
            "generator": "stprim.data.video_io.synthetic_tube",
            "frames": args.frames,
            "height": args.size,
            "width": args.size,
        }
    else:
        source_path = Path(args.video)
        video = load_video(
            source_path,
            max_frames=args.frames,
            start_frame=args.start_frame,
            resize=args.size,
        )
        source_name = str(source_path.resolve())
        fingerprint = source_fingerprint(source_path)
    seeds = args.seeds or [args.seed if args.seed is not None else 0]
    device = torch.device(args.device)
    target = video.to(device)
    environment = environment_report(device)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "frames": args.frames,
        "size": args.size,
        "start_frame": args.start_frame,
        "steps": sorted(set(args.steps)),
        "constraints": sorted(set(args.constraints)),
        "num_init": args.num_init,
        "max_primitives": args.max_primitives,
        "voxels": args.voxels,
        "support_capacity": args.support_capacity,
        "support_point_chunk": args.support_point_chunk,
        "adapt_every": args.adapt_every,
    }
    report_path = output_dir / "report.json"
    records: list[dict] = []
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if (
            existing.get("schema") != "temporal-tilt-ablation-v2"
            or existing.get("source") != source_name
            or existing.get("source_fingerprint") != fingerprint
            or existing.get("shape") != list(target.shape)
            or existing.get("protocol") != protocol
        ):
            raise ValueError("existing report does not match this experiment protocol")
        records = list(existing["records"])
    completed = {
        (
            record["geometry_constraint"],
            int(record["steps"]),
            int(record.get("seed", 0)),
        )
        for record in records
    }

    for seed in seeds:
        for steps in sorted(set(args.steps)):
            for constraint in args.constraints:
                key = (constraint, steps, seed)
                checkpoint = output_dir / (
                    f"{constraint}_seed{seed}_steps{steps}.pt"
                )
                if key in completed and checkpoint.exists():
                    print(f"skipping completed {key}", flush=True)
                    continue
                cfg = FitConfig(
                    num_init=args.num_init,
                    max_primitives=args.max_primitives,
                    steps=steps,
                    voxels_per_step=args.voxels,
                    cull_mode="support",
                    support_capacity=args.support_capacity,
                    support_point_chunk=args.support_point_chunk,
                    geometry_constraint=constraint,
                    seed=seed,
                    adapt_every=args.adapt_every,
                    log_every=max(1, steps // 5),
                )
                print(
                    f"running geometry={constraint} steps={steps} seed={seed}",
                    flush=True,
                )
                field, info = fit_volume(
                    target, cfg, device=args.device, verbose=False
                )
                rendered = render_field(field, info, cfg, mode="support")
                storage = field_storage(field, target)
                support_psnr = psnr(rendered, target)
                record = {
                    "geometry_constraint": constraint,
                    "steps": steps,
                    "seed": seed,
                    "environment": environment,
                    "voxel_evaluations": steps * args.voxels,
                    "fit_seconds": info["seconds"],
                    "n_final": info["n_final"],
                    "support_eval_psnr_db": support_psnr,
                    "psnr_db_per_1000_primitives": (
                        support_psnr / (info["n_final"] / 1000.0)
                    ),
                    "psnr_db_per_parameter_megabyte": (
                        support_psnr / storage["parameter_megabytes"]
                    ),
                    "storage": storage,
                    "error": reconstruction_error(rendered, target),
                    "structure": field_structure(
                        field,
                        frames=target.shape[0],
                        t_scale=cfg.t_scale,
                        support_sigma=cfg.support_sigma,
                    ),
                }
                records.append(record)
                completed.add(key)
                torch.save(
                    {"state": field.state_dict(), "cfg": vars(cfg), "info": info},
                    checkpoint,
                )
                report = {
                    "schema": "temporal-tilt-ablation-v2",
                    "source": source_name,
                    "source_fingerprint": fingerprint,
                    "environment": environment,
                    "shape": list(target.shape),
                    "protocol": protocol,
                    "records": records,
                    "summary": summarize(records),
                }
                write_report(report_path, report)
                print(
                    f"  PSNR={record['support_eval_psnr_db']:.2f} dB "
                    f"N={record['n_final']} time={record['fit_seconds']:.1f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
