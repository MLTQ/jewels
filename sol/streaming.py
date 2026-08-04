"""Persistent jewel lifecycles and rolling carry/commit windows."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.splat_density import temporal_standard_deviation


@dataclass(frozen=True)
class JewelLifecycles:
    """Finite-support lifecycles for globally identified jewels."""

    global_ids: torch.Tensor
    support_start_frames: torch.Tensor
    support_stop_frames: torch.Tensor
    first_active_frames: torch.Tensor
    last_active_frames: torch.Tensor
    active_frame_counts: torch.Tensor
    initial_mask: torch.Tensor

    @property
    def valid_mask(self) -> torch.Tensor:
        return self.active_frame_counts > 0


@dataclass(frozen=True)
class RollingWindow:
    """One prefix-conditioned commit step over a continuous fitted field."""

    index: int
    context_start: int
    frontier: int
    commit_stop: int
    context_ids: torch.Tensor
    carried_ids: torch.Tensor
    birth_ids: torch.Tensor
    active_commit_ids: torch.Tensor

    @property
    def view_start(self) -> int:
        return self.context_start

    @property
    def view_stop(self) -> int:
        return self.commit_stop


def frame_times(
    frames: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the fitted field's normalized timestamp for each physical frame."""
    if frames < 2:
        raise ValueError("streaming requires at least two frames")
    return torch.linspace(-1.0, 1.0, frames, device=device, dtype=dtype)


def normalized_time_to_frame(time: torch.Tensor, frames: int) -> torch.Tensor:
    """Map monolithic normalized timestamps into continuous physical-frame units."""
    if frames < 2:
        raise ValueError("streaming requires at least two frames")
    return (time + 1.0) * ((frames - 1) / 2.0)


def measure_jewel_lifecycles(
    features: torch.Tensor,
    frames: int,
    *,
    support_sigma: float = 3.0,
) -> JewelLifecycles:
    """Assign stable IDs and discrete lifecycles under finite temporal support."""
    if features.ndim != 2 or features.shape[1] != 22:
        raise ValueError("features must have shape (jewels, 22)")
    if support_sigma <= 0:
        raise ValueError("support_sigma must be positive")
    times = frame_times(frames, device=features.device, dtype=features.dtype)
    sigma = temporal_standard_deviation(features)
    radius = support_sigma * sigma
    support_start = normalized_time_to_frame(features[:, 2] - radius, frames)
    support_stop = normalized_time_to_frame(features[:, 2] + radius, frames)
    activity = (times[:, None] - features[None, :, 2]).abs() <= radius[None]
    active_counts = activity.sum(dim=0)
    valid = active_counts > 0
    first = activity.to(torch.int64).argmax(dim=0)
    last = frames - 1 - activity.flip(0).to(torch.int64).argmax(dim=0)
    first = torch.where(valid, first, torch.full_like(first, -1))
    last = torch.where(valid, last, torch.full_like(last, -1))
    return JewelLifecycles(
        global_ids=torch.arange(features.shape[0], device=features.device),
        support_start_frames=support_start,
        support_stop_frames=support_stop,
        first_active_frames=first,
        last_active_frames=last,
        active_frame_counts=active_counts,
        initial_mask=valid & (first == 0),
    )


def build_rolling_windows(
    lifecycles: JewelLifecycles,
    frames: int,
    *,
    prefix_frames: int,
    stride_frames: int,
) -> list[RollingWindow]:
    """Partition a clip into clamped-prefix and newly committed future strides."""
    if frames < 2:
        raise ValueError("streaming requires at least two frames")
    if prefix_frames < 0:
        raise ValueError("prefix_frames must be non-negative")
    if stride_frames <= 0:
        raise ValueError("stride_frames must be positive")
    first = lifecycles.first_active_frames
    last = lifecycles.last_active_frames
    valid = lifecycles.valid_mask
    windows = []
    for index, frontier in enumerate(range(0, frames, stride_frames)):
        commit_stop = min(frames, frontier + stride_frames)
        context_start = max(0, frontier - prefix_frames)
        active_commit = valid & (first < commit_stop) & (last >= frontier)
        carried = active_commit & (first < frontier)
        births = active_commit & (first >= frontier)
        context = (
            valid
            & (frontier > context_start)
            & (first < frontier)
            & (last >= context_start)
        )
        active_ids = lifecycles.global_ids[active_commit]
        carried_ids = lifecycles.global_ids[carried]
        birth_ids = lifecycles.global_ids[births]
        partition = torch.cat((carried_ids, birth_ids)).sort().values
        if not torch.equal(partition, active_ids):
            raise RuntimeError("carried and birth jewels must partition the commit state")
        windows.append(
            RollingWindow(
                index=index,
                context_start=context_start,
                frontier=frontier,
                commit_stop=commit_stop,
                context_ids=lifecycles.global_ids[context],
                carried_ids=carried_ids,
                birth_ids=birth_ids,
                active_commit_ids=active_ids,
            )
        )
    return windows
