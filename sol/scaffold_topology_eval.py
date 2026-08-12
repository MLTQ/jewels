"""Metrics and causal controls for scaffold-conditioned birth topology."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.scaffold_topology import ScaffoldTopologyModel, ScaffoldTopologyOutput


@dataclass(frozen=True)
class TopologyControlView:
    source_id: str
    class_id: int
    class_name: str
    index: int
    guide_raster: torch.Tensor
    carry_raster: torch.Tensor
    target_counts: torch.Tensor


def expand_topology_counts(
    counts: torch.Tensor, *, slots_per_cell: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand a dense cell-count field into canonical cell/rank index vectors."""
    if counts.ndim != 1 or counts.dtype != torch.long:
        raise ValueError("counts must be a one-dimensional int64 tensor")
    if slots_per_cell <= 0 or (counts < 0).any():
        raise ValueError("topology counts/capacity must be non-negative and valid")
    if len(counts) and int(counts.max()) > slots_per_cell:
        raise ValueError("topology count exceeds the declared per-cell capacity")
    cell_indices = torch.repeat_interleave(
        torch.arange(len(counts), device=counts.device), counts
    )
    if not len(cell_indices):
        return cell_indices.long(), cell_indices.long()
    starts = torch.cumsum(counts, dim=0) - counts
    slot_indices = torch.arange(
        len(cell_indices), device=counts.device
    ) - torch.repeat_interleave(starts, counts)
    return cell_indices.long(), slot_indices.long()


def topology_metrics(
    predicted_counts: list[torch.Tensor], target_counts: list[torch.Tensor]
) -> dict[str, float]:
    """Aggregate count, occupancy, slot-overlap, and spatial-correlation metrics."""
    if not predicted_counts or len(predicted_counts) != len(target_counts):
        raise ValueError("predicted and target count lists must be non-empty and aligned")
    predicted = torch.cat([values.float().reshape(-1) for values in predicted_counts])
    target = torch.cat([values.float().reshape(-1) for values in target_counts])
    if predicted.shape != target.shape or (predicted < 0).any() or (target < 0).any():
        raise ValueError("count tensors must have matching non-negative shapes")
    predicted_occupied = predicted > 0
    target_occupied = target > 0
    true_positive = (predicted_occupied & target_occupied).sum().float()
    false_positive = (predicted_occupied & ~target_occupied).sum().float()
    false_negative = (~predicted_occupied & target_occupied).sum().float()
    precision = true_positive / (true_positive + false_positive).clamp_min(1)
    recall = true_positive / (true_positive + false_negative).clamp_min(1)
    occupancy_f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    occupancy_iou = true_positive / (
        true_positive + false_positive + false_negative
    ).clamp_min(1)

    common = torch.minimum(predicted, target).sum()
    predicted_total = predicted.sum()
    target_total = target.sum()
    slot_precision = common / predicted_total.clamp_min(1)
    slot_recall = common / target_total.clamp_min(1)
    slot_f1 = 2 * slot_precision * slot_recall / (
        slot_precision + slot_recall
    ).clamp_min(1e-8)
    correlations = []
    for candidate, reference in zip(predicted_counts, target_counts, strict=True):
        candidate = candidate.float().reshape(-1)
        reference = reference.float().reshape(-1)
        candidate_centered = candidate - candidate.mean()
        reference_centered = reference - reference.mean()
        denominator = candidate_centered.norm() * reference_centered.norm()
        if float(denominator) > 1e-8:
            correlations.append(
                float((candidate_centered * reference_centered).sum() / denominator)
            )
    return {
        "cell_count_mae": float((predicted - target).abs().mean()),
        "cell_log_count_mae": float(
            (torch.log1p(predicted) - torch.log1p(target)).abs().mean()
        ),
        "total_count_ratio": float(predicted_total / target_total.clamp_min(1)),
        "occupancy_precision": float(precision),
        "occupancy_recall": float(recall),
        "occupancy_f1": float(occupancy_f1),
        "occupancy_iou": float(occupancy_iou),
        "slot_precision": float(slot_precision),
        "slot_recall": float(slot_recall),
        "slot_f1": float(slot_f1),
        "count_correlation": (
            sum(correlations) / len(correlations) if correlations else 0.0
        ),
        "predicted_births": float(predicted_total),
        "target_births": float(target_total),
    }


def decode_topology_counts(
    output: ScaffoldTopologyOutput,
    *,
    slots_per_cell: int,
    occupancy_threshold: float,
) -> torch.Tensor:
    """Decode without requiring the originating model instance."""
    if slots_per_cell <= 0 or not 0 < occupancy_threshold < 1:
        raise ValueError("decode capacity/threshold are invalid")
    occupied = torch.sigmoid(output.occupancy_logits) >= occupancy_threshold
    positive = output.positive_counts.round().long().clamp(1, slots_per_cell)
    return torch.where(occupied, positive, torch.zeros_like(positive))


def calibrate_occupancy_threshold(
    outputs: list[ScaffoldTopologyOutput],
    targets: list[torch.Tensor],
    *,
    slots_per_cell: int,
) -> tuple[float, dict[str, float]]:
    """Select a threshold on training views by slot F1, then count MAE."""
    if not outputs or len(outputs) != len(targets):
        raise ValueError("calibration outputs and targets must align")
    best = None
    for index in range(1, 20):
        threshold = index / 20
        predicted = [
            decode_topology_counts(
                output,
                slots_per_cell=slots_per_cell,
                occupancy_threshold=threshold,
            ).cpu()
            for output in outputs
        ]
        metrics = topology_metrics(predicted, [target.cpu() for target in targets])
        score = (metrics["slot_f1"], -metrics["cell_count_mae"], threshold)
        if best is None or score > best[0]:
            best = (score, threshold, metrics)
    assert best is not None
    return best[1], best[2]


@torch.no_grad()
def evaluate_topology_controls(
    model: ScaffoldTopologyModel,
    views: list[TopologyControlView],
    mean_counts_by_index: dict[int, torch.Tensor],
    *,
    occupancy_threshold: float,
    device: str | torch.device,
) -> dict:
    """Compare correct, class-shuffled, null, no-carry, and train-mean controls."""
    if not views:
        raise ValueError("topology evaluation requires at least one view")
    target_device = torch.device(device)
    by_index: dict[int, list[TopologyControlView]] = {}
    for view in views:
        by_index.setdefault(view.index, []).append(view)
    shuffled: dict[tuple[str, int], TopologyControlView] = {}
    for index, group in by_index.items():
        ordered = sorted(group, key=lambda item: (item.class_id, item.source_id))
        if len({item.class_id for item in ordered}) < 2:
            raise ValueError("each stride needs at least two classes for shuffled controls")
        for offset, view in enumerate(ordered):
            candidate = ordered[(offset + 1) % len(ordered)]
            if candidate.class_id == view.class_id:
                raise ValueError("shuffled scaffold must come from another class")
            shuffled[(view.source_id, index)] = candidate

    names = ("correct", "shuffled", "null", "correct_no_carry", "train_mean")
    predictions = {name: [] for name in names}
    targets = []
    classes: dict[str, dict[str, list[torch.Tensor]]] = {}
    model.eval()
    for view in sorted(views, key=lambda item: (item.source_id, item.index)):
        guide = view.guide_raster.to(target_device)
        carry = view.carry_raster.to(target_device)
        alternate = shuffled[(view.source_id, view.index)].guide_raster.to(target_device)
        outputs = {
            "correct": model(guide, carry),
            "shuffled": model(alternate, carry),
            "null": model(torch.zeros_like(guide), carry),
            "correct_no_carry": model(guide, torch.zeros_like(carry)),
        }
        for name, output in outputs.items():
            predictions[name].append(
                model.decode_counts(
                    output, occupancy_threshold=occupancy_threshold
                ).cpu()
            )
        mean = mean_counts_by_index.get(view.index)
        if mean is None:
            raise ValueError(f"missing train-mean baseline for stride {view.index}")
        predictions["train_mean"].append(mean.round().long().cpu())
        target = view.target_counts.cpu()
        targets.append(target)
        class_bucket = classes.setdefault(
            view.class_name, {name: [] for name in names}
        )
        for name in names:
            class_bucket[name].append(predictions[name][-1])
        class_bucket.setdefault("target", []).append(target)

    aggregate = {
        name: topology_metrics(values, targets) for name, values in predictions.items()
    }
    per_class = {
        class_name: {
            name: topology_metrics(values[name], values["target"])
            for name in names
        }
        for class_name, values in classes.items()
    }
    return {
        "occupancy_threshold": occupancy_threshold,
        "validation_views": len(views),
        "aggregate": aggregate,
        "per_class": per_class,
    }
