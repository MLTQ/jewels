"""Discrete local spacetime block language above continuous Jewel phrases."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from sol.jewel_casting_language import CastingNormalizer, _assign
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class BlockTokenCodebook:
    """Frozen descriptor normalization and prototypes for one block vocabulary."""

    prototypes: torch.Tensor
    descriptor_mean: torch.Tensor
    descriptor_std: torch.Tensor
    intrinsic_mean: torch.Tensor
    intrinsic_std: torch.Tensor
    block_shape: tuple[int, int, int]
    local_hist_shape: tuple[int, int, int]

    @property
    def vocabulary_size(self) -> int:
        return int(len(self.prototypes))

    @property
    def descriptor_dim(self) -> int:
        return int(self.prototypes.shape[1])

    def state_dict(self) -> dict:
        return {
            "prototypes": self.prototypes.cpu(),
            "descriptor_mean": self.descriptor_mean.cpu(),
            "descriptor_std": self.descriptor_std.cpu(),
            "intrinsic_mean": self.intrinsic_mean.cpu(),
            "intrinsic_std": self.intrinsic_std.cpu(),
            "block_shape": self.block_shape,
            "local_hist_shape": self.local_hist_shape,
        }

    @classmethod
    def from_state_dict(
        cls, state: dict, device: torch.device | str = "cpu"
    ) -> "BlockTokenCodebook":
        return cls(
            prototypes=state["prototypes"].to(device),
            descriptor_mean=state["descriptor_mean"].to(device),
            descriptor_std=state["descriptor_std"].to(device),
            intrinsic_mean=state["intrinsic_mean"].to(device),
            intrinsic_std=state["intrinsic_std"].to(device),
            block_shape=tuple(state["block_shape"]),
            local_hist_shape=tuple(state["local_hist_shape"]),
        )


def block_local_coordinates(
    centers: torch.Tensor, spec: GridSpec
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raster block IDs and continuous coordinates in each block's [-1,1]^3."""
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (N,3)")
    scaled = (centers.clamp(-1, 1) + 1) * 0.5
    shape = centers.new_tensor(spec.shape)
    block_coordinates = torch.floor(scaled * shape).long()
    limits = torch.tensor(spec.shape, device=centers.device) - 1
    block_coordinates = torch.minimum(block_coordinates, limits)
    local = (scaled * shape - block_coordinates.to(centers)) * 2 - 1
    cells = (
        (block_coordinates[:, 0] * spec.shape[1] + block_coordinates[:, 1])
        * spec.shape[2]
        + block_coordinates[:, 2]
    )
    return cells, local


def block_centers(spec: GridSpec, *, device: torch.device | str) -> torch.Tensor:
    """Return normalized continuous centers for raster-ordered blocks."""
    ids = torch.arange(spec.n_cells, device=device)
    t = ids % spec.shape[2]
    y = (ids // spec.shape[2]) % spec.shape[1]
    x = ids // (spec.shape[1] * spec.shape[2])
    coordinates = torch.stack([x, y, t], dim=1).float()
    shape = coordinates.new_tensor(spec.shape)
    return ((coordinates + 0.5) / shape) * 2 - 1


def block_serialization_order(spec: GridSpec) -> torch.Tensor:
    """Return time-major block IDs with Morton/Z order inside each spatial slab."""
    x_size, y_size, t_size = spec.shape
    if x_size & (x_size - 1) or y_size & (y_size - 1):
        raise ValueError("Morton block order requires power-of-two spatial dimensions")

    def morton(x: int, y: int) -> int:
        value = 0
        for bit in range(max(x_size, y_size).bit_length()):
            value |= ((x >> bit) & 1) << (2 * bit)
            value |= ((y >> bit) & 1) << (2 * bit + 1)
        return value

    output = []
    xy = sorted(
        ((x, y) for x in range(x_size) for y in range(y_size)),
        key=lambda pair: morton(*pair),
    )
    for t in range(t_size):
        for x, y in xy:
            output.append((x * y_size + y) * t_size + t)
    return torch.tensor(output, dtype=torch.long)


def block_descriptors(
    features: torch.Tensor,
    *,
    spec: GridSpec,
    intrinsic_mean: torch.Tensor,
    intrinsic_std: torch.Tensor,
    local_hist_shape: tuple[int, int, int] = (4, 4, 2),
) -> torch.Tensor:
    """Summarize every block without snapping or discarding any Jewel."""
    if features.ndim != 2 or features.shape[1] != 22:
        raise ValueError("block descriptors require Jewel features shaped (N,22)")
    if len(local_hist_shape) != 3 or any(value <= 0 for value in local_hist_shape):
        raise ValueError("local histogram shape must contain three positive values")
    if not torch.isfinite(features).all():
        raise ValueError("block descriptor features must be finite")
    cells, local = block_local_coordinates(features[:, :3], spec)
    counts = torch.bincount(cells, minlength=spec.n_cells).to(features)
    occupancy = torch.log1p(counts) / math.log1p(len(features))

    hist_shape = features.new_tensor(local_hist_shape)
    local_scaled = ((local + 1) * 0.5).clamp(0, 1 - 1e-7)
    bins = torch.floor(local_scaled * hist_shape).long()
    hist_size = math.prod(local_hist_shape)
    local_bin = (
        (bins[:, 0] * local_hist_shape[1] + bins[:, 1])
        * local_hist_shape[2]
        + bins[:, 2]
    )
    local_histogram = torch.bincount(
        cells * hist_size + local_bin,
        minlength=spec.n_cells * hist_size,
    ).reshape(spec.n_cells, hist_size).to(features)
    local_histogram = local_histogram / counts[:, None].clamp_min(1)

    normalized_intrinsic = (
        features[:, 3:] - intrinsic_mean.to(features)
    ) / intrinsic_std.to(features).clamp_min(1e-6)
    values = torch.cat([local, normalized_intrinsic], dim=1)
    sums = features.new_zeros(spec.n_cells, values.shape[1])
    squares = torch.zeros_like(sums)
    sums.scatter_add_(0, cells[:, None].expand_as(values), values)
    squares.scatter_add_(0, cells[:, None].expand_as(values), values.square())
    means = sums / counts[:, None].clamp_min(1)
    standard_deviations = (
        squares / counts[:, None].clamp_min(1) - means.square()
    ).clamp_min(0).sqrt()
    return torch.cat(
        [occupancy[:, None], local_histogram, means, standard_deviations], dim=1
    )


def _fit_lloyd(
    values: torch.Tensor,
    *,
    vocabulary_size: int,
    iterations: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if vocabulary_size <= 1 or vocabulary_size > len(values) or iterations <= 0:
        raise ValueError("block vocabulary fit settings are invalid")
    generator = torch.Generator(device=values.device).manual_seed(seed)
    centers = values[
        torch.randperm(len(values), generator=generator, device=values.device)[
            :vocabulary_size
        ]
    ].clone()
    for _ in range(iterations):
        assignments, _ = _assign(values, centers, chunk=1024)
        sums = values.new_zeros(centers.shape)
        counts = values.new_zeros(vocabulary_size)
        sums.scatter_add_(0, assignments[:, None].expand_as(values), values)
        counts.scatter_add_(
            0, assignments, torch.ones_like(assignments, dtype=values.dtype)
        )
        occupied = counts > 0
        centers[occupied] = sums[occupied] / counts[occupied, None]
        if (~occupied).any():
            replacements = torch.randperm(
                len(values), generator=generator, device=values.device
            )[: int((~occupied).sum())]
            centers[~occupied] = values[replacements]
    assignments, distances = _assign(values, centers, chunk=1024)
    counts = torch.bincount(assignments, minlength=vocabulary_size).float()
    probabilities = counts / counts.sum()
    nonzero = probabilities > 0
    perplexity = torch.exp(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
    return centers, {
        "mean_squared_assignment_distance": float(distances.mean()),
        "utilized_fraction": float((counts > 0).float().mean()),
        "perplexity": float(perplexity),
    }


def fit_block_token_codebook(
    fields: list[torch.Tensor],
    *,
    normalizer: CastingNormalizer,
    spec: GridSpec = GridSpec((8, 8, 4), slots_per_cell=1),
    local_hist_shape: tuple[int, int, int] = (4, 4, 2),
    vocabulary_size: int = 256,
    iterations: int = 20,
    seed: int = 20260905,
) -> tuple[BlockTokenCodebook, dict]:
    """Fit a training-owned discrete vocabulary over full local block summaries."""
    if not fields:
        raise ValueError("block vocabulary requires at least one field")
    descriptors = torch.cat(
        [
            block_descriptors(
                field,
                spec=spec,
                intrinsic_mean=normalizer.intrinsic_mean,
                intrinsic_std=normalizer.intrinsic_std,
                local_hist_shape=local_hist_shape,
            )
            for field in fields
        ]
    )
    mean = descriptors.mean(dim=0)
    std = descriptors.std(dim=0).clamp_min(1e-4)
    normalized = (descriptors - mean) / std
    prototypes, report = _fit_lloyd(
        normalized,
        vocabulary_size=vocabulary_size,
        iterations=iterations,
        seed=seed,
    )
    codebook = BlockTokenCodebook(
        prototypes=prototypes,
        descriptor_mean=mean,
        descriptor_std=std,
        intrinsic_mean=normalizer.intrinsic_mean.to(descriptors),
        intrinsic_std=normalizer.intrinsic_std.to(descriptors),
        block_shape=spec.shape,
        local_hist_shape=local_hist_shape,
    )
    return codebook, {
        **report,
        "training_fields": len(fields),
        "training_blocks": int(len(descriptors)),
        "descriptor_dim": int(descriptors.shape[1]),
        "vocabulary_size": vocabulary_size,
    }


def encode_block_tokens(
    features: torch.Tensor, codebook: BlockTokenCodebook
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign one frozen discrete token to every routing block in a field."""
    spec = GridSpec(codebook.block_shape, slots_per_cell=1)
    descriptors = block_descriptors(
        features,
        spec=spec,
        intrinsic_mean=codebook.intrinsic_mean,
        intrinsic_std=codebook.intrinsic_std,
        local_hist_shape=codebook.local_hist_shape,
    )
    normalized = (
        descriptors - codebook.descriptor_mean.to(descriptors)
    ) / codebook.descriptor_std.to(descriptors)
    tokens, distances = _assign(
        normalized, codebook.prototypes.to(normalized), chunk=1024
    )
    return tokens, distances


def most_frequent_block_token(programs: torch.Tensor, vocabulary_size: int) -> int:
    """Return the deterministic prompt-blind block token."""
    if programs.ndim != 2 or programs.numel() == 0:
        raise ValueError("block programs must have shape (fields,blocks)")
    return int(torch.bincount(programs.flatten(), minlength=vocabulary_size).argmax())
