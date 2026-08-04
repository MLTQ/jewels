"""Measurements and render-equivalence checks for persistent jewel streaming."""

from __future__ import annotations

import torch

from sol.render import render_truncated
from sol.splat_density import measure_frame_splat_density, summarize_counts
from sol.streaming import (
    JewelLifecycles,
    RollingWindow,
    build_rolling_windows,
    frame_times,
    measure_jewel_lifecycles,
)


def _quantile_summary(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    if not values.numel():
        return {key: 0.0 for key in ("mean", "p50", "p95", "max")}
    return {
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.5)),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _active_counts(lifecycles: JewelLifecycles, frames: int) -> torch.Tensor:
    indices = torch.arange(frames, device=lifecycles.global_ids.device)[:, None]
    return (
        lifecycles.valid_mask[None]
        & (lifecycles.first_active_frames[None] <= indices)
        & (lifecycles.last_active_frames[None] >= indices)
    ).sum(dim=1)


def measure_streaming_contract(
    features: torch.Tensor,
    shape: tuple[int, int, int],
    *,
    fps: float,
    prefix_frames: int,
    stride_frames: int,
    support_sigma: float = 3.0,
) -> tuple[dict, JewelLifecycles, list[RollingWindow]]:
    """Report density, births, lifespans, and rolling-state partitions."""
    frames, height, width = shape
    if fps <= 0:
        raise ValueError("fps must be positive")
    lifecycles = measure_jewel_lifecycles(
        features, frames, support_sigma=support_sigma
    )
    windows = build_rolling_windows(
        lifecycles,
        frames,
        prefix_frames=prefix_frames,
        stride_frames=stride_frames,
    )
    density = measure_frame_splat_density(
        features, frames, support_sigma=support_sigma
    )
    raw_active = _active_counts(lifecycles, frames)
    valid_lifespans = lifecycles.active_frame_counts[lifecycles.valid_mask]
    observed_jewels = int(lifecycles.valid_mask.sum())
    initial_jewels = int(lifecycles.initial_mask.sum())
    continuation_births = observed_jewels - initial_jewels
    megapixels = height * width / 1_000_000
    duration_seconds = frames / fps
    observed_birth_rate = observed_jewels / frames
    mean_lifespan = float(valid_lifespans.float().mean()) if observed_jewels else 0.0
    little_rhs = observed_birth_rate * mean_lifespan
    birth_counts = torch.bincount(
        lifecycles.first_active_frames[lifecycles.valid_mask], minlength=frames
    )
    effective_per_mp = density.effective_peak_alpha_counts / megapixels
    report = {
        "shape": list(shape),
        "fps": fps,
        "duration_seconds": duration_seconds,
        "support_sigma": support_sigma,
        "prefix_frames": prefix_frames,
        "stride_frames": stride_frames,
        "total_jewels": int(features.shape[0]),
        "observed_jewels": observed_jewels,
        "initial_jewels": initial_jewels,
        "continuation_births": continuation_births,
        "continuation_births_per_second": continuation_births / duration_seconds,
        "continuation_births_per_megapixel_second": (
            continuation_births / duration_seconds / megapixels
        ),
        "active_jewels_per_frame": summarize_counts(raw_active),
        "effective_jewels_per_frame": summarize_counts(
            density.effective_peak_alpha_counts
        ),
        "effective_jewels_per_megapixel_frame": summarize_counts(effective_per_mp),
        "lifespan_frames": _quantile_summary(valid_lifespans),
        "birth_counts_by_frame": birth_counts.cpu().tolist(),
        "effective_counts_by_frame": density.effective_peak_alpha_counts.cpu().tolist(),
        "little_law": {
            "mean_active_lhs": float(raw_active.float().mean()),
            "observed_births_per_frame": observed_birth_rate,
            "mean_observed_lifespan_frames": mean_lifespan,
            "birth_rate_times_lifespan_rhs": little_rhs,
            "absolute_error": abs(float(raw_active.float().mean()) - little_rhs),
        },
        "windows": [
            {
                "index": window.index,
                "context_frames": [window.context_start, window.frontier],
                "commit_frames": [window.frontier, window.commit_stop],
                "context_jewels": int(window.context_ids.numel()),
                "carried_jewels": int(window.carried_ids.numel()),
                "birth_jewels": int(window.birth_ids.numel()),
                "active_commit_jewels": int(window.active_commit_ids.numel()),
            }
            for window in windows
        ],
    }
    return report, lifecycles, windows


@torch.no_grad()
def audit_carry_commit_render(
    features: torch.Tensor,
    frames: int,
    windows: list[RollingWindow],
    *,
    support_sigma: float = 3.0,
    points_per_frame: int = 4,
    seed: int = 0,
) -> dict[str, float | int]:
    """Compare monolithic rendering with carried/born active subsets."""
    if points_per_frame <= 0:
        raise ValueError("points_per_frame must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    spatial = torch.rand(
        frames, points_per_frame, 2, generator=generator, dtype=features.dtype
    ) * 2 - 1
    times = frame_times(frames, dtype=features.dtype)
    points = torch.cat(
        (spatial, times[:, None, None].expand(-1, points_per_frame, 1)), dim=2
    ).reshape(-1, 3).to(features.device)
    point_frames = torch.arange(frames, device=features.device).repeat_interleave(
        points_per_frame
    )
    reference = render_truncated(
        features, points, support_sigma=support_sigma
    )
    streamed = torch.empty_like(reference)
    coverage = torch.zeros(points.shape[0], dtype=torch.int64, device=features.device)
    for window in windows:
        selected = (point_frames >= window.frontier) & (
            point_frames < window.commit_stop
        )
        streamed[selected] = render_truncated(
            features[window.active_commit_ids],
            points[selected],
            support_sigma=support_sigma,
        )
        coverage[selected] += 1
    error = (streamed - reference).abs()
    return {
        "points": int(points.shape[0]),
        "missing_points": int((coverage == 0).sum()),
        "duplicate_points": int((coverage > 1).sum()),
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
    }
