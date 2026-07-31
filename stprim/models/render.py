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


def cull_knn(points: torch.Tensor, mu: torch.Tensor, k: int) -> torch.Tensor:
    """Select the k nearest primitives (by Euclidean center distance).

    points (M, 3), mu (N, 3) -> (M, k) long.

    Euclidean rather than Mahalanobis on purpose: culling only needs a safe
    superset, and computing the full metric for all N would defeat the point.
    Far primitives have astronomically negative logits, so excluding them is
    numerically free.
    """
    k = min(k, mu.shape[0])
    d2 = torch.cdist(points, mu)
    return d2.topk(k, dim=1, largest=False).indices


def render_points(
    field: PrimitiveField,
    points: torch.Tensor,
    *,
    knn: int = 64,
    background: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate the field at arbitrary (M, 3) query points -> (M, 3) RGB.

    Works on a flat point list, not a grid, so the same path serves full-volume
    reconstruction and random-voxel stochastic training.
    """
    idx = cull_knn(points, field.mu.detach(), knn)  # (M, K)
    p = field.gather(idx)

    d = points[:, None, :] - p["mu"]  # (M, K, 3)

    # y = S^-1 R^T d  ->  q = |y|^2. Avoids materializing (M,K,3,3) inverses.
    rt_d = torch.einsum("mkji,mkj->mki", p["rot"], d)
    y = rt_d / (p["scale"] + 1e-8)
    q = (y * y).sum(-1)  # (M, K) squared Mahalanobis

    logits = -0.5 * q + torch.nn.functional.logsigmoid(p["logit_w"])

    color = p["color"]
    if field.p1_color:
        # P1 element: linear color ramp within the primitive's support.
        color = color + torch.einsum("mkij,mkj->mki", p["color_grad"], d)

    out = (logits.exp()[..., None] * color).sum(1)
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
