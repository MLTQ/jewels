"""Fixed train-only PCA codec for non-overlapping dense latent blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.latent_hierarchy import block_vectors, reshape_latents


@dataclass
class BlockPCACodec:
    mean: torch.Tensor
    components: torch.Tensor
    block_size: int
    grid_shape: tuple[int, int, int]
    latent_dim: int
    explained_variance: float

    def __post_init__(self) -> None:
        raw_dimension = self.block_size**3 * self.latent_dim
        if self.mean.shape != (raw_dimension,):
            raise ValueError("block mean has the wrong shape")
        if self.components.ndim != 2 or self.components.shape[1] != raw_dimension:
            raise ValueError("PCA components have the wrong shape")
        if any(axis % self.block_size for axis in self.grid_shape):
            raise ValueError("block size must divide the grid")

    @property
    def code_dim(self) -> int:
        return int(self.components.shape[0])

    @property
    def coarse_shape(self) -> tuple[int, int, int]:
        return tuple(axis // self.block_size for axis in self.grid_shape)

    def encode(self, latents: torch.Tensor) -> torch.Tensor:
        volume = reshape_latents(latents, self.grid_shape)
        vectors = block_vectors(volume, self.block_size)
        mean = self.mean.to(vectors)
        components = self.components.to(vectors)
        codes = (vectors - mean) @ components.T
        return codes.reshape(latents.shape[0], -1, self.code_dim)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        coarse_cells = self.coarse_shape[0] * self.coarse_shape[1] * self.coarse_shape[2]
        if codes.ndim != 3 or codes.shape[1:] != (coarse_cells, self.code_dim):
            raise ValueError("codes do not match the codec's coarse raster")
        components = self.components.to(codes)
        mean = self.mean.to(codes)
        vectors = codes.reshape(-1, self.code_dim) @ components + mean
        samples = codes.shape[0]
        cu, cv, ct = self.coarse_shape
        b = self.block_size
        blocked = vectors.reshape(
            samples, cu, cv, ct, b, b, b, self.latent_dim
        )
        volume = blocked.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(
            samples, *self.grid_shape, self.latent_dim
        )
        return volume.reshape(samples, -1, self.latent_dim)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean,
            "components": self.components,
            "block_size": self.block_size,
            "grid_shape": self.grid_shape,
            "latent_dim": self.latent_dim,
            "explained_variance": self.explained_variance,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "BlockPCACodec":
        return cls(
            mean=state["mean"].float(),
            components=state["components"].float(),
            block_size=int(state["block_size"]),
            grid_shape=tuple(state["grid_shape"]),
            latent_dim=int(state["latent_dim"]),
            explained_variance=float(state["explained_variance"]),
        )


def fit_block_pca(
    train_latents: torch.Tensor,
    grid_shape: tuple[int, int, int],
    *,
    block_size: int = 2,
    code_dim: int = 96,
    max_blocks: int = 100_000,
) -> BlockPCACodec:
    """Fit a deterministic PCA basis from training-source block vectors only."""
    if code_dim <= 0 or max_blocks <= 0:
        raise ValueError("code dimension and max blocks must be positive")
    volume = reshape_latents(train_latents, grid_shape)
    vectors = block_vectors(volume, block_size).float()
    raw_dimension = vectors.shape[1]
    if code_dim > raw_dimension:
        raise ValueError("code dimension exceeds raw block dimension")
    if len(vectors) > max_blocks:
        picks = torch.linspace(0, len(vectors) - 1, max_blocks).round().long()
        vectors = vectors[picks]
    mean = vectors.mean(dim=0)
    centered = vectors - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    components = eigenvectors[:, order[:code_dim]].T.contiguous()
    explained = float(eigenvalues[:code_dim].sum() / eigenvalues.sum().clamp_min(1e-12))
    return BlockPCACodec(
        mean=mean,
        components=components,
        block_size=block_size,
        grid_shape=grid_shape,
        latent_dim=train_latents.shape[-1],
        explained_variance=explained,
    )
