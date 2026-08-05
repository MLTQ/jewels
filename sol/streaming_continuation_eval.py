"""Correct/shuffled/null and rendered-field evaluation for jewel continuation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sol.render import render_exact
from sol.streaming import frame_times
from sol.streaming_data import (
    BirthTarget,
    ContinuationDataset,
    ContinuationView,
    rasterize_context,
)
from sol.streaming_features import to_global_time
from sol.streaming_model import BirthContinuationModel


@dataclass(frozen=True)
class ContinuationEvaluation:
    correct_feature_mse: float
    shuffled_feature_mse: float
    null_feature_mse: float
    correct_count_mae: float
    shuffled_count_mae: float
    null_count_mae: float
    predicted_count_ratio: float
    correct_render_psnr: float
    shuffled_render_psnr: float
    null_render_psnr: float
    carried_max_error: float

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


def _device_target(
    view: ContinuationView,
    dataset: ContinuationDataset,
    device: torch.device,
) -> BirthTarget:
    return BirthTarget(
        values=dataset.birth_standardizer.normalize(view.births.values).to(device),
        cell_indices=view.births.cell_indices.to(device),
        slot_indices=view.births.slot_indices.to(device),
        counts=view.births.counts.to(device),
        global_ids=view.births.global_ids.to(device),
        birth_frames=view.births.birth_frames.to(device),
    )


def _context_raster(
    view: ContinuationView,
    dataset: ContinuationDataset,
    device: torch.device,
) -> torch.Tensor:
    return rasterize_context(
        view.context_features,
        dataset.context_standardizer,
        prefix_frames=dataset.prefix_frames,
        stride_frames=dataset.stride_frames,
        grid_shape=dataset.grid_spec.shape,
    ).to(device)


def _render_points(
    total_frames: int,
    frontier: int,
    commit_stop: int,
    points_per_frame: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    frames = commit_stop - frontier
    spatial = torch.rand(frames, points_per_frame, 2, generator=generator) * 2 - 1
    times = frame_times(total_frames)[frontier:commit_stop]
    points = torch.cat(
        (spatial, times[:, None, None].expand(-1, points_per_frame, 1)), dim=2
    )
    return points.reshape(-1, 3).to(device)


def _nonoverlapping_context_index(
    views: tuple[ContinuationView, ...] | list[ContinuationView],
    index: int,
    prefix_frames: int,
) -> int:
    """Choose a wrong context that contains none of the evaluated target stride."""
    target = views[index]
    target_interval = (target.frontier, target.commit_stop)
    candidates = [
        (index - offset) % len(views) for offset in range(1, len(views))
    ]
    for candidate_index in candidates:
        candidate = views[candidate_index]
        context_interval = (candidate.frontier - prefix_frames, candidate.frontier)
        overlaps = max(target_interval[0], context_interval[0]) < min(
            target_interval[1], context_interval[1]
        )
        if not overlaps:
            return candidate_index
    raise ValueError("shuffled control requires a context disjoint from each target stride")


def _field_psnr(
    predicted_normalized: torch.Tensor,
    view: ContinuationView,
    dataset: ContinuationDataset,
    points: torch.Tensor,
) -> float:
    predicted_local = dataset.birth_standardizer.denormalize(predicted_normalized)
    predicted_global = to_global_time(
        predicted_local,
        dataset.total_frames,
        view.frontier,
        dataset.stride_frames,
    )
    candidate = torch.cat(
        (view.carried_global_features.to(points), predicted_global), dim=0
    )
    target = view.target_active_global_features.to(points)
    reference_rgb = render_exact(target, points)
    candidate_rgb = render_exact(candidate, points)
    mse = F.mse_loss(candidate_rgb, reference_rgb)
    return float(-10 * torch.log10(mse.clamp_min(1e-10)))


@torch.no_grad()
def evaluate_continuation(
    model: BirthContinuationModel,
    dataset: ContinuationDataset,
    *,
    device: str | torch.device,
    points_per_frame: int = 8,
    seed: int = 0,
) -> ContinuationEvaluation:
    """Evaluate whether the learned future depends on the correct prefix."""
    if points_per_frame <= 0:
        raise ValueError("points_per_frame must be positive")
    device = torch.device(device)
    model.eval()
    views = dataset.views
    targets = [_device_target(view, dataset, device) for view in views]
    contexts = [model.encode_context(_context_raster(view, dataset, device)) for view in views]
    feature_errors = {key: [] for key in ("correct", "shuffled", "null")}
    count_errors = {key: [] for key in ("correct", "shuffled", "null")}
    render_scores = {key: [] for key in ("correct", "shuffled", "null")}
    predicted_total = 0
    target_total = 0
    carried_max_error = 0.0
    for index, (view, target) in enumerate(zip(views, targets, strict=True)):
        condition_map = {
            "correct": contexts[index],
            "shuffled": contexts[
                _nonoverlapping_context_index(
                    views, index, dataset.prefix_frames
                )
            ],
            "null": torch.zeros_like(contexts[index]),
        }
        points = _render_points(
            dataset.total_frames,
            view.frontier,
            view.commit_stop,
            points_per_frame,
            seed + index,
            device,
        )
        for name, context in condition_map.items():
            output = model.forward_from_context(context, target)
            feature_errors[name].append(
                float(F.mse_loss(output.occupied_features, target.values))
                if len(target.values)
                else 0.0
            )
            predicted_counts = output.log_count.clamp_min(0).expm1()
            count_errors[name].append(
                float(F.l1_loss(predicted_counts, target.counts.float()))
            )
            render_scores[name].append(
                _field_psnr(output.occupied_features, view, dataset, points)
            )
        predicted = model.decode(_context_raster(view, dataset, device))
        predicted_total += int(predicted.counts.sum())
        target_total += len(view.births.values)
        carried = view.carried_global_features.to(device)
        carried_max_error = max(
            carried_max_error,
            float((carried - view.carried_global_features.to(device)).abs().max()),
        )

    def average(values: list[float]) -> float:
        return sum(values) / len(values)

    return ContinuationEvaluation(
        correct_feature_mse=average(feature_errors["correct"]),
        shuffled_feature_mse=average(feature_errors["shuffled"]),
        null_feature_mse=average(feature_errors["null"]),
        correct_count_mae=average(count_errors["correct"]),
        shuffled_count_mae=average(count_errors["shuffled"]),
        null_count_mae=average(count_errors["null"]),
        predicted_count_ratio=predicted_total / max(target_total, 1),
        correct_render_psnr=average(render_scores["correct"]),
        shuffled_render_psnr=average(render_scores["shuffled"]),
        null_render_psnr=average(render_scores["null"]),
        carried_max_error=carried_max_error,
    )
