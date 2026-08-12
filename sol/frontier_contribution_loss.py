"""Differentiable visible-contribution supervision at each emission frontier."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sol.splat_density import temporal_standard_deviation


@dataclass(frozen=True)
class FrontierContributionLoss:
    """Per-jewel and cell-aggregate frontier contribution losses."""

    total: torch.Tensor
    per_jewel: torch.Tensor
    cell_alpha: torch.Tensor
    visible_count: torch.Tensor


def frontier_peak_alpha(
    features: torch.Tensor, *, covariance_chunk: int = 4096
) -> torch.Tensor:
    """Return each local jewel's best possible alpha at frontier time zero."""
    if features.ndim != 2 or features.shape[1] != 22:
        raise ValueError("features must have shape (jewels,22)")
    if covariance_chunk <= 0:
        raise ValueError("covariance chunk must be positive")
    if not len(features):
        return features.new_empty(0)
    sigma = torch.cat(
        [
            temporal_standard_deviation(features[start : start + covariance_chunk])
            for start in range(0, len(features), covariance_chunk)
        ]
    ).clamp_min(1e-8).detach()
    standardized = features[:, 2].abs() / sigma
    return torch.sigmoid(features[:, 21]) * torch.exp(-0.5 * standardized.square())


def _pool(values: torch.Tensor, cells: torch.Tensor, n_cells: int) -> torch.Tensor:
    output = values.new_zeros(n_cells)
    output.scatter_add_(0, cells, values)
    return output


def frontier_contribution_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    cell_indices: torch.Tensor,
    *,
    n_cells: int,
    visible_threshold: float = 0.05,
    visible_temperature: float = 0.01,
    covariance_chunk: int = 4096,
) -> FrontierContributionLoss:
    """Match frontier alpha, cell alpha mass, and soft visible-jewel counts."""
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("predicted and target marks must have matching matrices")
    if cell_indices.shape != (len(predicted),) or cell_indices.dtype != torch.long:
        raise ValueError("cell indices must be one int64 value per jewel")
    if n_cells <= 0 or (len(cell_indices) and int(cell_indices.max()) >= n_cells):
        raise ValueError("cell indices exceed the declared raster")
    if not 0 < visible_threshold < 1 or visible_temperature <= 0:
        raise ValueError("visible threshold/temperature are invalid")
    predicted_alpha = frontier_peak_alpha(
        predicted, covariance_chunk=covariance_chunk
    )
    with torch.no_grad():
        target_alpha = frontier_peak_alpha(
            target, covariance_chunk=covariance_chunk
        )
    epsilon = predicted_alpha.new_tensor(1e-8)
    per_jewel = F.smooth_l1_loss(
        (predicted_alpha + epsilon).sqrt(),
        (target_alpha + epsilon).sqrt(),
    )
    predicted_cell_alpha = _pool(predicted_alpha, cell_indices, n_cells)
    target_cell_alpha = _pool(target_alpha, cell_indices, n_cells)
    cell_alpha = F.smooth_l1_loss(
        torch.log1p(predicted_cell_alpha), torch.log1p(target_cell_alpha)
    )
    predicted_visible = torch.sigmoid(
        (predicted_alpha - visible_threshold) / visible_temperature
    )
    target_visible = torch.sigmoid(
        (target_alpha - visible_threshold) / visible_temperature
    )
    visible_count = F.smooth_l1_loss(
        torch.log1p(_pool(predicted_visible, cell_indices, n_cells)),
        torch.log1p(_pool(target_visible, cell_indices, n_cells)),
    )
    return FrontierContributionLoss(
        per_jewel + cell_alpha + visible_count,
        per_jewel,
        cell_alpha,
        visible_count,
    )
