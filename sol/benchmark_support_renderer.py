"""Benchmark support-complete tiled rendering against the production KNN path.

The source is a fitted checkpoint, not a randomly initialized field. Each
requested primitive budget is a deterministic random subset of that fitted
geometry, and query points are sampled from the checkpoint's recorded volume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
STPRIM_ROOT = ROOT / "stprim"
if str(STPRIM_ROOT) not in sys.path:
    sys.path.insert(0, str(STPRIM_ROOT))

from core.params import PrimitiveField  # noqa: E402
from core.volume import make_grid  # noqa: E402
from models.render import (  # noqa: E402
    render_points,
    support_aabb_half_extent,
)
from models.tiled_support import (  # noqa: E402
    build_support_tile_index,
    query_support_pairs,
)


def file_sha256(path: Path) -> str:
    """Return a source fingerprint suitable for an experiment report."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subset_field(
    state: dict[str, torch.Tensor],
    count: int,
    *,
    seed: int,
    device: torch.device,
) -> PrimitiveField:
    """Build a deterministic fitted-geometry subset on the benchmark device."""
    available = int(state["mu"].shape[0])
    if not 0 < count <= available:
        raise ValueError(f"primitive count {count} is outside 1..{available}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = torch.randperm(available, generator=generator)[:count]
    p1_color = "color_grad" in state
    field = PrimitiveField(count, p1_color=p1_color, device=device)
    field.load_state_dict({name: value[selected].to(device) for name, value in state.items()})
    return field


def candidate_statistics(
    field: PrimitiveField,
    points: torch.Tensor,
    *,
    support_sigma: float,
    capacity: int,
    base_resolution: int,
    level_scale: float,
) -> dict[str, float | int]:
    """Measure true conservative-sphere pair density for a query sample."""
    with torch.no_grad():
        index = build_support_tile_index(
            field.mu,
            field.scales().amax(dim=1),
            half_extent=support_aabb_half_extent(
                field.scales(),
                field.rotations(),
                support_sigma=support_sigma,
            ),
            metric_scale=field.scales(),
            metric_rotation=field.rotations(),
            support_sigma=support_sigma,
            base_resolution=base_resolution,
            level_scale=level_scale,
        )
        owners, _ = query_support_pairs(index, points, capacity=capacity)
        counts = torch.bincount(owners, minlength=points.shape[0]).float()
    return {
        "pair_count": int(owners.numel()),
        "candidates_mean": float(counts.mean()),
        "candidates_p50": float(counts.quantile(0.5)),
        "candidates_p90": float(counts.quantile(0.9)),
        "candidates_max": int(counts.max()),
    }


def timed_training_step(
    field: PrimitiveField,
    points: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str,
    repeats: int,
    warmup: int,
    knn: int,
    support_sigma: float,
    capacity: int,
    point_chunk: int,
    base_resolution: int,
    level_scale: float,
) -> dict[str, float | int]:
    """Time forward, scalar MSE, and backward with synchronized CUDA events."""
    kwargs = {
        "knn": knn,
        "cull_mode": mode,
        "support_sigma": support_sigma,
        "support_capacity": capacity,
        "support_point_chunk": point_chunk,
        "support_base_resolution": base_resolution,
        "support_level_scale": level_scale,
    }

    def run_once() -> None:
        field.zero_grad(set_to_none=True)
        prediction = render_points(field, points, **kwargs)
        torch.nn.functional.mse_loss(prediction, target).backward()

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    seconds = []
    peak_bytes = []
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_once()
        end.record()
        torch.cuda.synchronize()
        seconds.append(start.elapsed_time(end) / 1000.0)
        peak_bytes.append(torch.cuda.max_memory_allocated())
    return {
        "seconds_median": statistics.median(seconds),
        "seconds_min": min(seconds),
        "peak_allocated_bytes_max": max(peak_bytes),
    }


def correctness_audit(
    field: PrimitiveField,
    points: torch.Tensor,
    *,
    support_sigma: float,
    capacity: int,
    point_chunk: int,
    base_resolution: int,
    level_scale: float,
) -> dict[str, float]:
    """Compare tiled output with the independent all-center support oracle."""
    with torch.no_grad():
        reference = render_points(
            field,
            points,
            cull_mode="support",
            support_sigma=support_sigma,
            support_capacity=capacity,
            support_point_chunk=point_chunk,
        )
        tiled = render_points(
            field,
            points,
            cull_mode="support_tiled",
            support_sigma=support_sigma,
            support_capacity=capacity,
            support_point_chunk=point_chunk,
            support_base_resolution=base_resolution,
            support_level_scale=level_scale,
        )
        difference = (reference - tiled).abs()
    return {
        "mean_abs": float(difference.mean()),
        "max_abs": float(difference.max()),
    }


def summarize(records: list[dict], *, ratio_gate: float) -> dict:
    """Evaluate the predeclared correctness, completion, and throughput gates."""
    completed = [record for record in records if "error" not in record]
    tiled = {record["primitive_count"]: record for record in completed if record["mode"] == "support_tiled"}
    knn = {record["primitive_count"]: record for record in completed if record["mode"] == "knn"}
    shared = sorted(set(tiled) & set(knn))
    ratios = {
        str(count): tiled[count]["seconds_median"] / knn[count]["seconds_median"]
        for count in shared
    }
    audits = [record["correctness"] for record in tiled.values() if "correctness" in record]
    return {
        "training_step_ratio_tiled_over_knn": ratios,
        "all_requested_modes_completed": len(completed) == len(records),
        "correctness_max_abs_below_2e_5": bool(
            audits and max(audit["max_abs"] for audit in audits) < 2e-5
        ),
        "within_ratio_gate_at_all_shared_scales": bool(
            ratios and max(ratios.values()) <= ratio_gate
        ),
        "ratio_gate": ratio_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--primitive-counts", type=int, nargs="+", default=[10000, 45000, 72000])
    parser.add_argument("--modes", nargs="+", choices=("knn", "support_tiled"), default=["knn", "support_tiled"])
    parser.add_argument("--queries", type=int, default=8192)
    parser.add_argument("--audit-queries", type=int, default=16)
    parser.add_argument("--candidate-queries", type=int, default=1024)
    parser.add_argument("--knn", type=int, default=64)
    parser.add_argument("--support-sigma", type=float, default=5.0)
    parser.add_argument("--support-capacity", type=int, default=16384)
    parser.add_argument("--support-point-chunk", type=int, default=256)
    parser.add_argument("--support-base-resolution", type=int, default=32)
    parser.add_argument("--support-level-scale", type=float, default=1.55)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ratio-gate", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("this benchmark requires CUDA for synchronized timing and memory")
    if args.repeats <= 0 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative")
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["state"]
    available = int(state["mu"].shape[0])
    counts = sorted(set(args.primitive_counts))
    if not counts or counts[-1] > available:
        raise ValueError(f"checkpoint has {available} primitives, requested {counts}")

    device = torch.device(args.device)
    shape = tuple(checkpoint["info"]["shape"])
    t_scale = float(checkpoint.get("cfg", {}).get("t_scale", 1.0))
    grid = make_grid(shape, t_scale=t_scale, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    query_indices = torch.randint(len(grid), (args.queries,), generator=generator)
    points = grid[query_indices].to(device)
    target = torch.rand(args.queries, 3, generator=generator).to(device)

    records = []
    started = time.time()
    for count in counts:
        field = subset_field(state, count, seed=args.seed, device=device)
        candidate_stats = candidate_statistics(
            field,
            points[: min(args.candidate_queries, len(points))],
            support_sigma=args.support_sigma,
            capacity=args.support_capacity,
            base_resolution=args.support_base_resolution,
            level_scale=args.support_level_scale,
        )
        audit = correctness_audit(
            field,
            points[: min(args.audit_queries, len(points))],
            support_sigma=args.support_sigma,
            capacity=args.support_capacity,
            point_chunk=args.support_point_chunk,
            base_resolution=args.support_base_resolution,
            level_scale=args.support_level_scale,
        )
        for mode in args.modes:
            print(f"benchmarking primitives={count} mode={mode}", flush=True)
            record = {
                "primitive_count": count,
                "mode": mode,
                "query_count": args.queries,
            }
            if mode == "support_tiled":
                record["candidate_statistics"] = candidate_stats
                record["correctness"] = audit
            try:
                record.update(
                    timed_training_step(
                        field,
                        points,
                        target,
                        mode=mode,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        knn=args.knn,
                        support_sigma=args.support_sigma,
                        capacity=args.support_capacity,
                        point_chunk=args.support_point_chunk,
                        base_resolution=args.support_base_resolution,
                        level_scale=args.support_level_scale,
                    )
                )
            except (RuntimeError, MemoryError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
                torch.cuda.empty_cache()
            records.append(record)

    report = {
        "schema": "support-renderer-benchmark-v1",
        "source": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "available_primitives": available,
            "shape": list(shape),
        },
        "protocol": {
            key: value
            for key, value in vars(args).items()
            if key not in {"checkpoint", "out", "device"}
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "device_uuid": str(torch.cuda.get_device_properties(device).uuid),
        },
        "elapsed_seconds": time.time() - started,
        "records": records,
        "summary": summarize(records, ratio_gate=args.ratio_gate),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
