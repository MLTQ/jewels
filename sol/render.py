"""Reference and conservative finite-support renderers for canonical jewels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


_TRIU = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def covariance_terms(
    features: torch.Tensor, *, eigen_chunk: int = 4096
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (covariance, precision) from canonical log-covariance features."""
    if features.ndim != 2 or features.shape[1] < 22:
        raise ValueError("features must have shape (N,22+)")
    if eigen_chunk <= 0:
        raise ValueError("eigen_chunk must be positive")
    if not len(features):
        empty = features.new_empty((0, 3, 3))
        return empty, empty.clone()
    covariances, precisions = [], []
    for start in range(0, len(features), eigen_chunk):
        part = features[start : start + eigen_chunk]
        log_sigma = part.new_zeros(part.shape[0], 3, 3)
        for offset, (i, j) in enumerate(_TRIU):
            value = part[:, 3 + offset]
            log_sigma[:, i, j] = value
            log_sigma[:, j, i] = value
        eigenvalues, eigenvectors = torch.linalg.eigh(log_sigma.float())
        eigenvalues = eigenvalues.clamp(-16.0, 4.0)
        covariances.append(
            torch.einsum(
                "nij,nj,nkj->nik", eigenvectors, eigenvalues.exp(), eigenvectors
            ).to(features.dtype)
        )
        precisions.append(
            torch.einsum(
                "nij,nj,nkj->nik",
                eigenvectors,
                (-eigenvalues).exp(),
                eigenvectors,
            ).to(features.dtype)
        )
    return torch.cat(covariances), torch.cat(precisions)


def _render_block(
    features: torch.Tensor,
    points: torch.Tensor,
    precision: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    support_sigma: float | None = None,
) -> torch.Tensor:
    centers = features[:, :3]
    delta = points[:, None, :] - centers[None, :, :]
    mahalanobis = torch.einsum("mni,nij,mnj->mn", delta, precision, delta)
    logits = -0.5 * mahalanobis + F.logsigmoid(features[:, 21])[None]
    if candidate_mask is not None:
        logits = logits.masked_fill(~candidate_mask, -torch.inf)
    if support_sigma is not None:
        logits = logits.masked_fill(mahalanobis > support_sigma**2, -torch.inf)
    color = features[:, 9:12][None] + torch.einsum(
        "nij,mnj->mni", features[:, 12:21].reshape(-1, 3, 3), delta
    )
    return (logits.exp()[..., None] * color).sum(dim=1)


def render_exact(
    features: torch.Tensor,
    points: torch.Tensor,
    *,
    background: torch.Tensor | None = None,
    point_chunk: int = 2048,
    primitive_chunk: int = 4096,
) -> torch.Tensor:
    """Evaluate every Gaussian at every point, chunked but otherwise untruncated."""
    _, precision = covariance_terms(features)
    outputs = []
    for point_start in range(0, points.shape[0], point_chunk):
        query = points[point_start : point_start + point_chunk]
        value = query.new_zeros(query.shape[0], 3)
        for prim_start in range(0, features.shape[0], primitive_chunk):
            part = slice(prim_start, prim_start + primitive_chunk)
            value += _render_block(features[part], query, precision[part])
        if background is not None:
            value += background.to(value)
        outputs.append(value)
    return torch.cat(outputs, dim=0)


def render_truncated(
    features: torch.Tensor,
    points: torch.Tensor,
    *,
    support_sigma: float = 5.0,
    background: torch.Tensor | None = None,
    point_chunk: int = 2048,
    primitive_chunk: int = 4096,
) -> torch.Tensor:
    """Render all jewels within an explicit Mahalanobis support radius.

    A world-axis AABB is used only as a conservative prefilter. Any point within
    the support ellipsoid must be inside this AABB, so unlike Euclidean center
    kNN it cannot omit an elongated jewel inside the declared support.
    """
    if support_sigma <= 0:
        raise ValueError("support_sigma must be positive")
    covariance, precision = covariance_terms(features)
    outputs = []
    for point_start in range(0, points.shape[0], point_chunk):
        query = points[point_start : point_start + point_chunk]
        value = query.new_zeros(query.shape[0], 3)
        for prim_start in range(0, features.shape[0], primitive_chunk):
            part = slice(prim_start, prim_start + primitive_chunk)
            part_features = features[part]
            extent = support_sigma * torch.diagonal(
                covariance[part], dim1=-2, dim2=-1
            ).clamp_min(0).sqrt()
            delta = query[:, None, :] - part_features[None, :, :3]
            candidate = (delta.abs() <= extent[None]).all(dim=-1)
            value += _render_block(
                part_features,
                query,
                precision[part],
                candidate_mask=candidate,
                support_sigma=support_sigma,
            )
        if background is not None:
            value += background.to(value)
        outputs.append(value)
    return torch.cat(outputs, dim=0)


def render_euclidean_knn(
    features: torch.Tensor,
    points: torch.Tensor,
    *,
    k: int,
) -> torch.Tensor:
    """Research baseline reproducing the unsafe center-kNN approximation."""
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, features.shape[0])
    _, precision = covariance_terms(features)
    indices = torch.cdist(points, features[:, :3]).topk(
        k, dim=1, largest=False
    ).indices
    outputs = []
    for row, selected in zip(points, indices, strict=True):
        outputs.append(
            _render_block(
                features[selected], row[None], precision[selected]
            )[0]
        )
    return torch.stack(outputs)


@dataclass(frozen=True)
class RenderAudit:
    support_sigma: float
    max_abs_error: float
    mean_abs_error: float


def audit_truncation(
    features: torch.Tensor,
    points: torch.Tensor,
    *,
    support_sigma: float = 5.0,
) -> RenderAudit:
    """Compare finite-support rendering against the all-jewel reference."""
    reference = render_exact(features, points)
    candidate = render_truncated(features, points, support_sigma=support_sigma)
    error = (candidate - reference).abs()
    return RenderAudit(
        support_sigma=support_sigma,
        max_abs_error=float(error.max()),
        mean_abs_error=float(error.mean()),
    )
