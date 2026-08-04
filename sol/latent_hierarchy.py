"""Quantitative hierarchy diagnostics for dense raster latent fields."""

from __future__ import annotations

import torch


def reshape_latents(
    latents: torch.Tensor, grid_shape: tuple[int, int, int]
) -> torch.Tensor:
    """Restore flat raster order to `(sample,u,v,t,dimension)`."""
    if latents.ndim != 3:
        raise ValueError("latents must have shape (samples,cells,dimension)")
    cells = grid_shape[0] * grid_shape[1] * grid_shape[2]
    if latents.shape[1] != cells:
        raise ValueError(f"grid has {cells} cells but latents have {latents.shape[1]}")
    return latents.reshape(latents.shape[0], *grid_shape, latents.shape[-1])


def axis_neighbor_correlations(volume: torch.Tensor) -> dict[str, float]:
    """Return scalar Pearson correlation between adjacent u/v/t cell latents."""
    if volume.ndim != 5:
        raise ValueError("volume must have shape (samples,u,v,t,dimension)")
    correlations = {}
    for name, axis in zip(("u", "v", "t"), (1, 2, 3), strict=True):
        before = volume.narrow(axis, 0, volume.shape[axis] - 1).float()
        after = volume.narrow(axis, 1, volume.shape[axis] - 1).float()
        before = before - before.mean()
        after = after - after.mean()
        denominator = before.square().mean().sqrt() * after.square().mean().sqrt()
        correlations[name] = float((before * after).mean() / denominator.clamp_min(1e-12))
    return correlations


def block_vectors(volume: torch.Tensor, block_size: int) -> torch.Tensor:
    """Flatten non-overlapping cubic blocks into independent row vectors."""
    if volume.ndim != 5 or block_size <= 0:
        raise ValueError("volume and block size are invalid")
    samples, gu, gv, gt, dimension = volume.shape
    if any(size % block_size for size in (gu, gv, gt)):
        raise ValueError("block size must divide every grid axis")
    blocked = volume.reshape(
        samples,
        gu // block_size,
        block_size,
        gv // block_size,
        block_size,
        gt // block_size,
        block_size,
        dimension,
    )
    return blocked.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
        -1, block_size**3 * dimension
    )


def hierarchy_report(
    latents: torch.Tensor,
    grid_shape: tuple[int, int, int],
    *,
    block_sizes: tuple[int, ...] = (2, 4),
    pca_block_size: int = 2,
    pca_dimensions: tuple[int, ...] = (24, 48, 64, 96),
    max_pca_blocks: int = 100_000,
) -> dict:
    """Measure local redundancy and linear block-code feasibility."""
    if max_pca_blocks <= 0:
        raise ValueError("max PCA blocks must be positive")
    volume = reshape_latents(latents, grid_shape)
    pooling = {}
    for size in block_sizes:
        vectors = block_vectors(volume, size).float().reshape(
            -1, size**3, latents.shape[-1]
        )
        mean = vectors.mean(dim=1, keepdim=True)
        pooling[str(size)] = {
            "coarse_grid": tuple(axis // size for axis in grid_shape),
            "blocks": int(vectors.shape[0]),
            "repeat_mean_mse": float((vectors - mean).square().mean()),
        }

    vectors = block_vectors(volume, pca_block_size).float()
    if len(vectors) > max_pca_blocks:
        picks = torch.linspace(0, len(vectors) - 1, max_pca_blocks).round().long()
        vectors = vectors[picks]
    vectors = vectors - vectors.mean(dim=0, keepdim=True)
    covariance = vectors.T @ vectors / max(len(vectors) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    total = eigenvalues.sum().clamp_min(1e-12)
    explained = {
        str(dimension): float(eigenvalues[:dimension].sum() / total)
        for dimension in pca_dimensions
        if 0 < dimension <= vectors.shape[1]
    }
    return {
        "samples": int(latents.shape[0]),
        "grid_shape": grid_shape,
        "latent_dim": int(latents.shape[-1]),
        "axis_neighbor_correlation": axis_neighbor_correlations(volume),
        "pooling": pooling,
        "pca": {
            "block_size": pca_block_size,
            "raw_block_dim": int(vectors.shape[1]),
            "sampled_blocks": int(vectors.shape[0]),
            "explained_variance": explained,
        },
    }
