"""Sequential oracle-mark rollout for predicted scaffold topology."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.scaffold_topology_data import ScaffoldTopologyView, rasterize_carried_state
from sol.scaffold_topology_eval import topology_metrics
from sol.splat_density import measure_frame_splat_density, summarize_counts
from sol.streaming import measure_jewel_lifecycles
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class OracleTopologyRollout:
    features: torch.Tensor
    global_ids: torch.Tensor
    report: dict


def oracle_matched_birth_mask(
    view: ScaffoldTopologyView, predicted_counts: torch.Tensor
) -> torch.Tensor:
    """Retain target marks only where the predicted cell/rank exists."""
    if predicted_counts.shape != view.births.counts.shape:
        raise ValueError("predicted counts do not match the topology grid")
    if (predicted_counts < 0).any():
        raise ValueError("predicted counts must be non-negative")
    return view.births.slot_indices < predicted_counts[view.births.cell_indices]


@torch.no_grad()
def rollout_oracle_matched_topology(
    model,
    views: tuple[ScaffoldTopologyView, ...],
    guides: list[torch.Tensor],
    target_features: torch.Tensor,
    total_frames: int,
    spec: GridSpec,
    *,
    stride_frames: int,
    support_sigma: float,
    occupancy_threshold: float,
    device: str | torch.device,
) -> OracleTopologyRollout:
    """Roll forward predicted counts while copying retained target marks and carried IDs."""
    if not views or len(views) != len(guides):
        raise ValueError("rollout views and guides must be non-empty and aligned")
    ordered = sorted(zip(views, guides, strict=True), key=lambda item: item[0].frontier)
    if ordered[0][0].frontier != 0:
        raise ValueError("rollout must begin at frontier zero")
    target_device = torch.device(device)
    model.eval()
    state_features = target_features.cpu().new_empty((0, 22))
    state_ids = torch.empty(0, dtype=torch.long)
    predicted_fields = []
    target_fields = []
    windows = []
    max_carry_error = 0.0

    for view, guide in ordered:
        if len(state_features):
            lifecycles = measure_jewel_lifecycles(
                state_features, total_frames, support_sigma=support_sigma
            )
            carry_mask = (
                lifecycles.valid_mask
                & (lifecycles.first_active_frames < view.frontier)
                & (lifecycles.last_active_frames >= view.frontier)
            )
            carried = state_features[carry_mask]
            carried_ids = state_ids[carry_mask]
        else:
            carried = state_features
            carried_ids = state_ids
        carry_snapshot = carried.clone()
        carry_raster = rasterize_carried_state(
            carried,
            total_frames,
            view.frontier,
            stride_frames,
            spec,
            support_sigma=support_sigma,
        )
        output = model(guide.to(target_device), carry_raster.to(target_device))
        predicted = model.decode_counts(
            output, occupancy_threshold=occupancy_threshold
        ).long().cpu()
        target = view.births.counts.cpu()
        matched = oracle_matched_birth_mask(view, predicted)
        new_features = view.birth_global_features.cpu()[matched]
        new_ids = view.births.global_ids.cpu()[matched]
        if torch.isin(new_ids, state_ids).any():
            raise RuntimeError("a rollout attempted to emit an existing stable ID")
        before = state_features.clone()
        state_features = torch.cat((state_features, new_features), dim=0)
        state_ids = torch.cat((state_ids, new_ids), dim=0)
        if len(before):
            max_carry_error = max(
                max_carry_error,
                float((state_features[: len(before)] - before).abs().max()),
            )
        if len(carry_snapshot):
            current = state_features[torch.isin(state_ids, carried_ids)]
            if current.shape != carry_snapshot.shape:
                raise RuntimeError("carried stable IDs changed cardinality")
            max_carry_error = max(
                max_carry_error,
                float((current - carry_snapshot).abs().max()),
            )
        predicted_fields.append(predicted.cpu())
        target_fields.append(target.cpu())
        common = int(torch.minimum(predicted, target).sum())
        windows.append(
            {
                "index": view.index,
                "frontier": view.frontier,
                "commit_stop": view.commit_stop,
                "carried_jewels": len(carried),
                "target_births": int(target.sum()),
                "predicted_births": int(predicted.sum()),
                "oracle_matched_births": common,
                "unmaterialized_false_positive_births": int(
                    (predicted - target).clamp_min(0).sum()
                ),
            }
        )

    completed_frames = ordered[-1][0].commit_stop
    density_features = state_features
    density_target = target_features.cpu()
    candidate_density = measure_frame_splat_density(
        density_features, total_frames, support_sigma=support_sigma
    )
    target_density = measure_frame_splat_density(
        density_target, total_frames, support_sigma=support_sigma
    )
    candidate_effective = candidate_density.effective_peak_alpha_counts[:completed_frames]
    target_effective = target_density.effective_peak_alpha_counts[:completed_frames]
    candidate_alpha = candidate_density.peak_alpha_counts[0.05][:completed_frames]
    target_alpha = target_density.peak_alpha_counts[0.05][:completed_frames]
    report = {
        "windows": windows,
        "completed_frames": completed_frames,
        "emitted_jewels": len(state_features),
        "stable_ids_unique": len(state_ids) == len(torch.unique(state_ids)),
        "max_carry_feature_error": max_carry_error,
        "topology": topology_metrics(predicted_fields, target_fields),
        "oracle_retained_density": {
            "effective": summarize_counts(candidate_effective),
            "target_effective": summarize_counts(target_effective),
            "effective_mean_ratio": float(
                candidate_effective.float().mean()
                / target_effective.float().mean().clamp_min(1e-8)
            ),
            "above_5_percent_alpha": summarize_counts(candidate_alpha),
            "target_above_5_percent_alpha": summarize_counts(target_alpha),
        },
        "oracle_mark_policy": (
            "Materialize only target marks whose predicted cell/rank exists; report but do not "
            "render false-positive ranks. Density is therefore an optimistic topology-recall bound."
        ),
    }
    return OracleTopologyRollout(state_features.cpu(), state_ids.cpu(), report)
