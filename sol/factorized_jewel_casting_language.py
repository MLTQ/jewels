"""Compositional role tokens for irregular spacetime Jewel casts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch

from sol.jewel_casting_language import (
    CastingBundles,
    CastingNormalizer,
    _assign,
    _cell_size,
    bundle_field,
)
from sol.token_grid import GridSpec


FACTOR_DIMENSIONS: dict[str, tuple[int, ...]] = {
    "layout": tuple(range(0, 3)),
    "covariance": tuple(range(3, 9)),
    "surface": (9, 10, 11, 21),
    "gradient": tuple(range(12, 21)),
}


@dataclass(frozen=True)
class FactorCodebook:
    """One physical role's prototypes and count-aware assignment coordinate."""

    name: str
    dimensions: tuple[int, ...]
    prototypes: torch.Tensor
    prototype_count_coordinates: torch.Tensor


@dataclass(frozen=True)
class FactorizedCodebook:
    """Four composable role vocabularies sharing one casting coordinate contract."""

    factors: tuple[FactorCodebook, ...]
    normalizer: CastingNormalizer
    grid_shape: tuple[int, int, int]
    bundle_size: int
    count_weight: float

    @property
    def vocabulary_size(self) -> int:
        sizes = {len(factor.prototypes) for factor in self.factors}
        if len(sizes) != 1:
            raise RuntimeError("factor vocabularies must share a size")
        return int(next(iter(sizes)))


@dataclass(frozen=True)
class FactorizedProgram:
    """A cast sequence with one discrete token per physical role."""

    cells: torch.Tensor
    anchors: torch.Tensor
    counts: torch.Tensor
    tokens: dict[str, torch.Tensor]
    residuals: torch.Tensor
    source_jewels: int

    @property
    def casts(self) -> int:
        return int(len(self.cells))

    @property
    def discrete_decisions(self) -> int:
        return self.casts * len(self.tokens)


def _factor_vectors(
    bundles: CastingBundles,
    dimensions: tuple[int, ...],
    count_weight: float,
) -> torch.Tensor:
    selected = bundles.values[:, :, dimensions].flatten(1)
    count = bundles.counts.to(selected).div(bundles.values.shape[1])
    return torch.cat([selected, count[:, None] * count_weight], dim=1)


def _fit_centers(
    values: torch.Tensor,
    *,
    vocabulary_size: int,
    iterations: int,
    assignment_chunk: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    initial = torch.randperm(
        len(values), generator=generator, device=values.device
    )[:vocabulary_size]
    centers = values[initial].clone()
    for _ in range(iterations):
        assignments, _ = _assign(values, centers, chunk=assignment_chunk)
        sums = values.new_zeros(centers.shape)
        counts = values.new_zeros(vocabulary_size)
        sums.scatter_add_(0, assignments[:, None].expand_as(values), values)
        counts.scatter_add_(
            0, assignments, torch.ones_like(assignments, dtype=values.dtype)
        )
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty, None]
        if (~nonempty).any():
            replacement = torch.randperm(
                len(values), generator=generator, device=values.device
            )[: int((~nonempty).sum())]
            centers[~nonempty] = values[replacement]
    assignments, distance = _assign(values, centers, chunk=assignment_chunk)
    counts = torch.bincount(assignments, minlength=vocabulary_size).float()
    probabilities = counts / counts.sum()
    entropy = -(
        probabilities[probabilities > 0] * probabilities[probabilities > 0].log()
    ).sum()
    return centers, {
        "mean_squared_assignment_distance": float(distance.mean()),
        "utilized_fraction": float((counts > 0).float().mean()),
        "perplexity": float(entropy.exp()),
    }


def fit_factorized_codebook(
    fields: list[torch.Tensor],
    *,
    spec: GridSpec,
    bundle_size: int,
    vocabulary_size: int,
    iterations: int = 15,
    max_casts: int = 100_000,
    assignment_chunk: int = 512,
    count_weight: float = 4.0,
    seed: int = 0,
) -> tuple[FactorizedCodebook, dict]:
    """Fit independent role codebooks over the same sampled canonical casts."""
    if vocabulary_size <= 1 or iterations <= 0 or max_casts <= 0:
        raise ValueError("factorized codebook hyperparameters are invalid")
    normalizer = CastingNormalizer.fit(fields)
    bundles = [
        bundle_field(
            field, spec=spec, bundle_size=bundle_size, normalizer=normalizer
        )
        for field in fields
    ]
    cast_count = sum(bundle.casts for bundle in bundles)
    generator = torch.Generator(device=fields[0].device).manual_seed(seed)
    if cast_count > max_casts:
        sample = torch.randperm(
            cast_count, generator=generator, device=fields[0].device
        )[:max_casts]
    else:
        sample = torch.arange(cast_count, device=fields[0].device)
    factors = []
    reports = {}
    for factor_index, (name, dimensions) in enumerate(FACTOR_DIMENSIONS.items()):
        values = torch.cat(
            [_factor_vectors(bundle, dimensions, count_weight) for bundle in bundles]
        )[sample]
        if len(values) < vocabulary_size:
            raise ValueError("vocabulary cannot exceed sampled casts")
        factor_generator = torch.Generator(device=values.device).manual_seed(
            seed + factor_index + 1
        )
        centers, factor_report = _fit_centers(
            values,
            vocabulary_size=vocabulary_size,
            iterations=iterations,
            assignment_chunk=assignment_chunk,
            generator=factor_generator,
        )
        factors.append(
            FactorCodebook(
                name=name,
                dimensions=dimensions,
                prototypes=centers[:, :-1].reshape(
                    vocabulary_size, bundle_size, len(dimensions)
                ),
                prototype_count_coordinates=centers[:, -1],
            )
        )
        reports[name] = factor_report
    return FactorizedCodebook(
        factors=tuple(factors),
        normalizer=normalizer,
        grid_shape=spec.shape,
        bundle_size=bundle_size,
        count_weight=count_weight,
    ), {"sampled_casts": len(sample), "factors": reports}


def composed_prototypes(
    program: FactorizedProgram, codebook: FactorizedCodebook
) -> torch.Tensor:
    """Compose independently selected physical roles into aligned Jewel bundles."""
    values = program.residuals.new_zeros(
        program.casts, codebook.bundle_size, 22
    )
    for factor in codebook.factors:
        dimensions = list(factor.dimensions)
        values[:, :, dimensions] = factor.prototypes.to(values)[
            program.tokens[factor.name]
        ]
    return values


def encode_factorized_program(
    features: torch.Tensor, codebook: FactorizedCodebook
) -> FactorizedProgram:
    """Assign every role token while retaining exact continuous residual targets."""
    spec = GridSpec(codebook.grid_shape, slots_per_cell=1)
    bundles = bundle_field(
        features,
        spec=spec,
        bundle_size=codebook.bundle_size,
        normalizer=codebook.normalizer,
    )
    tokens = {}
    composed = bundles.values.new_zeros(bundles.values.shape)
    for factor in codebook.factors:
        vectors = _factor_vectors(bundles, factor.dimensions, codebook.count_weight)
        prototypes = torch.cat(
            [
                factor.prototypes.flatten(1).to(vectors),
                factor.prototype_count_coordinates.to(vectors)[:, None],
            ],
            dim=1,
        )
        token, _ = _assign(vectors, prototypes, chunk=2048)
        tokens[factor.name] = token
        composed[:, :, list(factor.dimensions)] = factor.prototypes.to(composed)[token]
    return FactorizedProgram(
        cells=bundles.cells,
        anchors=bundles.anchors,
        counts=bundles.counts,
        tokens=tokens,
        residuals=bundles.values - composed,
        source_jewels=bundles.source_jewels,
    )


def decode_factorized_program(
    program: FactorizedProgram,
    codebook: FactorizedCodebook,
    *,
    residual_scale: float = 1.0,
) -> torch.Tensor:
    """Decode a compositional cast phrase into canonical continuous Jewel features."""
    if not math.isfinite(residual_scale):
        raise ValueError("residual scale must be finite")
    values = composed_prototypes(program, codebook) + residual_scale * program.residuals
    spec = GridSpec(codebook.grid_shape, slots_per_cell=1)
    size = _cell_size(spec, values)
    mean = codebook.normalizer.intrinsic_mean.to(values)
    std = codebook.normalizer.intrinsic_std.to(values)
    output = []
    for index, count in enumerate(program.counts.tolist()):
        part = values[index, :count]
        decoded = values.new_empty(count, 22)
        decoded[:, :3] = program.anchors[index] + part[:, :3] * size
        decoded[:, 3:] = part[:, 3:] * std + mean
        output.append(decoded)
    result = torch.cat(output)
    if len(result) != program.source_jewels:
        raise RuntimeError("factorized casting decode changed the Jewel count")
    return result


def factor_histograms(
    program: FactorizedProgram, *, n_cells: int, vocabulary_size: int
) -> dict[str, torch.Tensor]:
    """Count each role's tokens per addressed cell."""
    histograms = {}
    for name, tokens in program.tokens.items():
        flat = program.cells * vocabulary_size + tokens
        histograms[name] = torch.bincount(
            flat,
            weights=program.counts.float(),
            minlength=n_cells * vocabulary_size,
        ).reshape(n_cells, vocabulary_size)
    return histograms


def load_factorized_codebook(
    path: str | Path, device: torch.device | str = "cpu"
) -> FactorizedCodebook:
    """Restore the portable role-codebook artifact written by casting audits."""
    saved = torch.load(path, map_location=device, weights_only=False)
    factors = tuple(
        FactorCodebook(
            name=name,
            dimensions=tuple(row["dimensions"]),
            prototypes=row["prototypes"].to(device),
            prototype_count_coordinates=row["prototype_count_coordinates"].to(device),
        )
        for name, row in saved["factors"].items()
    )
    return FactorizedCodebook(
        factors=factors,
        normalizer=CastingNormalizer(
            saved["normalizer"]["intrinsic_mean"].to(device),
            saved["normalizer"]["intrinsic_std"].to(device),
        ),
        grid_shape=tuple(saved["grid_shape"]),
        bundle_size=int(saved["bundle_size"]),
        count_weight=float(saved["count_weight"]),
    )
