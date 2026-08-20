"""Renderer for spacetime primitive fields.

Additive anisotropic Gaussian splats over a learned constant background. Given
Mahalanobis distance q_i(x) = (x-mu_i)^T Sigma_i^-1 (x-mu_i):

    value(x) = bg + sum_i exp(-q_i/2 + log w_i) * c_i(x)

where c_i is constant (P0) or a linear ramp (P1) per primitive.

A soft-Voronoi normalization of the same logits (softmax instead of sum) was
implemented and A/B'd against this in 2026-07, including a steelman round
(background pseudo-cell + Lloyd relaxation). It lost reconstruction AND
canonicality on real footage and was removed — read the PROJECT.md decision
log before resurrecting the idea; the with-voronoi tree is archived at
jewels/stprim-final-with-voronoi-20260731.tar.gz.
"""

from __future__ import annotations

import torch

from core.params import PrimitiveField
from models.tiled_support import (
    SupportOverflowError,
    build_support_tile_index,
    query_support_pairs,
)


def cull_knn(
    points: torch.Tensor, mu: torch.Tensor, k: int, *, chunk: int = 16384
) -> torch.Tensor:
    """Select the k nearest primitives (by Euclidean center distance).

    points (M, 3), mu (N, 3) -> (M, k) long.

    This is the historical fast path. It is not support-safe for anisotropic
    primitives: an elongated primitive can contribute far from its center and
    still be omitted by center-distance KNN. Use ``cull_mode="support"`` when
    correctness of the candidate set matters.

    Chunked over points: identical indices, but peak memory is chunk x N
    instead of M x N — the full 65536 x 10000 distance matrix was the
    fitter's entire 13 GB VRAM spike.
    """
    k = min(k, mu.shape[0])
    # `torch.cdist` materializes roughly chunk×N distances. The original fixed
    # 16k chunk is safe at 10k primitives but requests ~3 GB at 45k, which can
    # fail late in densification on an 8 GB card. Preserve the caller's upper
    # bound while capping the distance workspace. A 100M-pair cap is the measured
    # throughput knee on the 2070S: the distance workspace stays below 0.5 GB
    # and the complete culler below 0.9 GB, while larger chunks do not improve
    # latency.
    max_distance_pairs = 100_000_000
    chunk = min(chunk, max(1024, max_distance_pairs // max(mu.shape[0], 1)))
    if points.shape[0] <= chunk:
        return torch.cdist(points, mu).topk(k, dim=1, largest=False).indices
    outs = []
    for i in range(0, points.shape[0], chunk):
        d2 = torch.cdist(points[i : i + chunk], mu)
        outs.append(d2.topk(k, dim=1, largest=False).indices)
    return torch.cat(outs, 0)


def cull_support_sphere(
    points: torch.Tensor,
    mu: torch.Tensor,
    max_scale: torch.Tensor,
    *,
    support_sigma: float = 5.0,
    capacity: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a provably complete finite-support candidate set.

    For every primitive, ``||x-mu|| / max(scale)`` is a lower bound on its
    Mahalanobis radius. Therefore every primitive inside ``support_sigma`` is
    inside this conservative sphere. The returned boolean mask removes loose
    sphere candidates after gathering. If more than ``capacity`` candidates
    can be inside the sphere, fail loudly instead of silently truncating.
    """
    if support_sigma <= 0:
        raise ValueError("support_sigma must be positive")
    if capacity <= 0:
        raise ValueError("support capacity must be positive")
    if mu.shape[0] == 0:
        raise ValueError("cannot render an empty primitive field")
    if max_scale.shape != (mu.shape[0],):
        raise ValueError("max_scale must have shape (num_primitives,)")

    capacity = min(capacity, mu.shape[0])
    probe_k = min(capacity + 1, mu.shape[0])
    normalized_distance = torch.cdist(points, mu) / max_scale.clamp_min(1e-8)[None]
    distance, idx = normalized_distance.topk(
        probe_k, dim=1, largest=False, sorted=True
    )
    if mu.shape[0] > capacity:
        overflow = distance[:, capacity] <= support_sigma
        if bool(overflow.any()):
            count = int(overflow.sum())
            raise SupportOverflowError(
                f"support candidate capacity {capacity} is insufficient for "
                f"{count}/{points.shape[0]} query points; increase "
                "support_capacity or reduce support_sigma"
            )

    return idx[:, :capacity], distance[:, :capacity] <= support_sigma


def _render_candidates(
    field: PrimitiveField,
    points: torch.Tensor,
    idx: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    support_sigma: float | None = None,
) -> torch.Tensor:
    """Evaluate already-selected candidates without adding a background."""
    p = field.gather(idx)

    d = points[:, None, :] - p["mu"]

    # y = S^-1 R^T d  ->  q = |y|^2. Avoids materializing (M,K,3,3) inverses.
    rt_d = torch.einsum("mkji,mkj->mki", p["rot"], d)
    y = rt_d / (p["scale"] + 1e-8)
    q = (y * y).sum(-1)

    logits = -0.5 * q + torch.nn.functional.logsigmoid(p["logit_w"])
    if support_sigma is not None:
        inside = q <= support_sigma * support_sigma
        if candidate_mask is not None:
            inside = inside & candidate_mask
        logits = logits.masked_fill(~inside, -torch.inf)

    color = p["color"]
    if field.p1_color:
        color = color + torch.einsum("mkij,mkj->mki", p["color_grad"], d)

    return (logits.exp()[..., None] * color).sum(1)


def _render_candidate_pairs(
    field: PrimitiveField,
    points: torch.Tensor,
    owners: torch.Tensor,
    primitive_indices: torch.Tensor,
    *,
    support_sigma: float,
) -> torch.Tensor:
    """Evaluate ragged query/primitive pairs and reduce them by query."""
    if owners.numel() == 0:
        return torch.zeros(
            points.shape[0], 3, dtype=points.dtype, device=points.device
        )
    p = field.gather(primitive_indices)
    d = points[owners] - p["mu"]
    rt_d = torch.einsum("pji,pj->pi", p["rot"], d)
    y = rt_d / (p["scale"] + 1e-8)
    q = y.square().sum(dim=-1)
    logits = -0.5 * q + torch.nn.functional.logsigmoid(p["logit_w"])
    logits = logits.masked_fill(q > support_sigma * support_sigma, -torch.inf)
    color = p["color"]
    if field.p1_color:
        color = color + torch.einsum("pij,pj->pi", p["color_grad"], d)
    contribution = logits.exp()[:, None] * color
    return torch.zeros(
        points.shape[0], 3, dtype=contribution.dtype, device=points.device
    ).index_add(0, owners, contribution)


def support_aabb_half_extent(
    scale: torch.Tensor,
    rotation: torch.Tensor,
    *,
    support_sigma: float,
) -> torch.Tensor:
    """World-axis AABB half extents for finite-support ellipsoids."""
    return support_sigma * torch.sqrt(
        (rotation.square() * scale[:, None, :].square()).sum(dim=2)
    )


def render_points(
    field: PrimitiveField,
    points: torch.Tensor,
    *,
    knn: int = 64,
    cull_mode: str = "knn",
    support_sigma: float = 5.0,
    support_capacity: int = 512,
    support_point_chunk: int = 4096,
    support_base_resolution: int = 32,
    support_level_scale: float = 1.55,
    background: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate the field at arbitrary (M, 3) query points -> (M, 3) RGB.

    Works on a flat point list, not a grid, so the same path serves full-volume
    reconstruction and random-voxel stochastic training.
    """
    if cull_mode == "knn":
        idx = cull_knn(points, field.mu.detach(), knn)
        out = _render_candidates(field, points, idx)
    elif cull_mode == "exact":
        idx = torch.arange(len(field), device=points.device).expand(
            points.shape[0], -1
        )
        out = _render_candidates(field, points, idx)
    elif cull_mode == "support":
        if support_point_chunk <= 0:
            raise ValueError("support_point_chunk must be positive")
        max_scale = field.scales().detach().amax(dim=1)
        outs = []
        for start in range(0, points.shape[0], support_point_chunk):
            query = points[start : start + support_point_chunk]
            idx, candidate_mask = cull_support_sphere(
                query,
                field.mu.detach(),
                max_scale,
                support_sigma=support_sigma,
                capacity=support_capacity,
            )
            outs.append(
                _render_candidates(
                    field,
                    query,
                    idx,
                    candidate_mask=candidate_mask,
                    support_sigma=support_sigma,
                )
            )
        out = torch.cat(outs, 0)
    elif cull_mode == "support_tiled":
        if support_point_chunk <= 0:
            raise ValueError("support_point_chunk must be positive")
        detached_scale = field.scales().detach()
        detached_rotation = field.rotations().detach()
        tile_index = build_support_tile_index(
            field.mu.detach(),
            detached_scale.amax(dim=1),
            half_extent=support_aabb_half_extent(
                detached_scale,
                detached_rotation,
                support_sigma=support_sigma,
            ),
            metric_scale=detached_scale,
            metric_rotation=detached_rotation,
            support_sigma=support_sigma,
            base_resolution=support_base_resolution,
            level_scale=support_level_scale,
        )
        outs = []
        for start in range(0, points.shape[0], support_point_chunk):
            query = points[start : start + support_point_chunk]
            owners, primitive_indices = query_support_pairs(
                tile_index, query, capacity=support_capacity
            )
            outs.append(
                _render_candidate_pairs(
                    field,
                    query,
                    owners,
                    primitive_indices,
                    support_sigma=support_sigma,
                )
            )
        out = torch.cat(outs, 0)
    else:
        raise ValueError(f"unknown cull_mode {cull_mode!r}")

    if background is not None:
        out = out + background
    return out


def render_volume(
    field: PrimitiveField,
    grid: torch.Tensor,
    *,
    chunk: int = 65536,
    **kw,
) -> torch.Tensor:
    """Render a full (M, 3) coordinate grid in chunks. Inference helper."""
    outs = []
    for i in range(0, grid.shape[0], chunk):
        outs.append(render_points(field, grid[i : i + chunk], **kw))
    return torch.cat(outs, 0)
