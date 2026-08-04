"""Per-frame effective-density measurements for canonical spacetime jewels."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FrameSplatDensity:
    """Raw per-frame density vectors retained for transparent aggregation."""

    times: torch.Tensor
    support_counts: torch.Tensor
    peak_alpha_counts: dict[float, torch.Tensor]
    effective_peak_alpha_counts: torch.Tensor


def temporal_standard_deviation(features: torch.Tensor) -> torch.Tensor:
    """Recover marginal temporal sigma from gauge-free log-covariance features."""
    if features.ndim != 2 or features.shape[1] != 22:
        raise ValueError("features must have shape (jewels, 22)")
    upper = torch.triu_indices(3, 3, device=features.device)
    log_covariance = features.new_zeros(features.shape[0], 3, 3)
    log_covariance[:, upper[0], upper[1]] = features[:, 3:9]
    log_covariance = log_covariance + log_covariance.transpose(1, 2)
    log_covariance -= torch.diag_embed(
        torch.diagonal(log_covariance, dim1=1, dim2=2) / 2
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(log_covariance.double())
    temporal_variance = (
        eigenvectors[:, 2].square() * eigenvalues.exp()
    ).sum(dim=1)
    return temporal_variance.clamp_min(1e-16).sqrt().to(features.dtype)


def measure_frame_splat_density(
    features: torch.Tensor,
    frames: int,
    *,
    support_sigma: float = 3.0,
    peak_alpha_thresholds: tuple[float, ...] = (0.01, 0.05),
) -> FrameSplatDensity:
    """Measure temporal support and potential peak contribution at each frame."""
    if frames <= 0 or support_sigma <= 0:
        raise ValueError("frames and support sigma must be positive")
    if not peak_alpha_thresholds or any(
        threshold <= 0 or threshold >= 1 for threshold in peak_alpha_thresholds
    ):
        raise ValueError("peak alpha thresholds must be non-empty and inside (0,1)")
    times = torch.linspace(-1, 1, frames, dtype=features.dtype, device=features.device)
    temporal_sigma = temporal_standard_deviation(features)
    standardized_time = (
        times[:, None] - features[None, :, 2]
    ).abs() / temporal_sigma[None]
    peak_alpha = torch.sigmoid(features[None, :, 21]) * torch.exp(
        -0.5 * standardized_time.square()
    )
    support_counts = (standardized_time <= support_sigma).sum(dim=1)
    threshold_counts = {
        threshold: (peak_alpha >= threshold).sum(dim=1)
        for threshold in peak_alpha_thresholds
    }
    alpha_sum = peak_alpha.sum(dim=1)
    effective = alpha_sum.square() / peak_alpha.square().sum(dim=1).clamp_min(1e-12)
    return FrameSplatDensity(times, support_counts, threshold_counts, effective)


def summarize_counts(values: torch.Tensor) -> dict[str, float]:
    """Produce JSON-safe distribution statistics for one per-frame count vector."""
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("count summary requires a non-empty vector")
    floating = values.float()
    return {
        "mean": float(floating.mean()),
        "median": float(floating.median()),
        "min": float(floating.min()),
        "max": float(floating.max()),
    }
