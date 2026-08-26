"""Discrete-continuous casting programs for irregular spacetime Jewels."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from sol.token_grid import GridSpec


INTRINSIC_DIMENSIONS = tuple(range(3, 22))


@dataclass(frozen=True)
class CastingNormalizer:
    """Train-owned feature moments for cell-relative motif coordinates."""

    intrinsic_mean: torch.Tensor
    intrinsic_std: torch.Tensor

    @classmethod
    def fit(cls, fields: list[torch.Tensor]) -> "CastingNormalizer":
        if not fields:
            raise ValueError("cannot fit a casting normalizer without fields")
        if any(field.ndim != 2 or field.shape[1] != 22 for field in fields):
            raise ValueError("casting fields must have shape (N,22)")
        intrinsic = torch.cat([field[:, 3:].double() for field in fields])
        return cls(
            intrinsic.mean(dim=0).float(),
            intrinsic.std(dim=0).clamp_min(1e-4).float(),
        )


@dataclass(frozen=True)
class CastingBundles:
    """Canonical local constellations before motif assignment."""

    cells: torch.Tensor
    anchors: torch.Tensor
    counts: torch.Tensor
    values: torch.Tensor
    source_jewels: int

    @property
    def casts(self) -> int:
        return int(len(self.cells))


@dataclass(frozen=True)
class MotifCodebook:
    """Learned bundle prototypes and their train-owned coordinate contract."""

    prototypes: torch.Tensor
    prototype_count_coordinates: torch.Tensor
    normalizer: CastingNormalizer
    grid_shape: tuple[int, int, int]
    bundle_size: int
    count_weight: float

    @property
    def vocabulary_size(self) -> int:
        return int(len(self.prototypes))


@dataclass(frozen=True)
class CastProgram:
    """A serialized sequence of motif casts plus continuous residual state."""

    cells: torch.Tensor
    anchors: torch.Tensor
    counts: torch.Tensor
    motifs: torch.Tensor
    residuals: torch.Tensor
    source_jewels: int

    @property
    def casts(self) -> int:
        return int(len(self.cells))


def _cell_size(spec: GridSpec, like: torch.Tensor) -> torch.Tensor:
    return like.new_tensor([2.0 / value for value in spec.shape])


def _lexicographic_center_order(features: torch.Tensor) -> torch.Tensor:
    order = torch.arange(len(features), device=features.device)
    for dimension in (2, 1, 0):
        rank = torch.argsort(features[order, dimension], stable=True)
        order = order[rank]
    return order


def bundle_field(
    features: torch.Tensor,
    *,
    spec: GridSpec,
    bundle_size: int,
    normalizer: CastingNormalizer,
) -> CastingBundles:
    """Serialize every Jewel into canonical cell-local constellation bundles."""
    if features.ndim != 2 or features.shape[1] != 22:
        raise ValueError("features must have shape (N,22)")
    if bundle_size <= 0 or not torch.isfinite(features).all():
        raise ValueError("bundle size and field values are invalid")
    cells = spec.cell_index(features[:, :3])
    size = _cell_size(spec, features)
    mean = normalizer.intrinsic_mean.to(features)
    std = normalizer.intrinsic_std.to(features)
    bundle_cells = []
    anchors = []
    counts = []
    values = []
    for cell in torch.unique(cells, sorted=True).tolist():
        selected = features[cells == cell]
        selected = selected[_lexicographic_center_order(selected)]
        for start in range(0, len(selected), bundle_size):
            part = selected[start : start + bundle_size]
            count = len(part)
            anchor = part[:, :3].mean(dim=0)
            normalized = features.new_zeros(bundle_size, 22)
            normalized[:count, :3] = (part[:, :3] - anchor) / size
            normalized[:count, 3:] = (part[:, 3:] - mean) / std
            bundle_cells.append(cell)
            anchors.append(anchor)
            counts.append(count)
            values.append(normalized)
    bundled = CastingBundles(
        cells=torch.as_tensor(bundle_cells, device=features.device).long(),
        anchors=torch.stack(anchors),
        counts=torch.as_tensor(counts, device=features.device).long(),
        values=torch.stack(values),
        source_jewels=len(features),
    )
    if int(bundled.counts.sum()) != len(features):
        raise RuntimeError("casting serialization lost target Jewels")
    return bundled


def _vectors(bundles: CastingBundles, count_weight: float) -> torch.Tensor:
    count = bundles.counts.to(bundles.values).div(bundles.values.shape[1])
    return torch.cat(
        [bundles.values.flatten(1), count[:, None] * count_weight], dim=1
    )


def _assign(
    vectors: torch.Tensor, centers: torch.Tensor, *, chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if chunk <= 0:
        raise ValueError("assignment chunk must be positive")
    assignments = []
    distances = []
    for start in range(0, len(vectors), chunk):
        distance = torch.cdist(vectors[start : start + chunk].float(), centers.float())
        nearest_distance, nearest = distance.min(dim=1)
        assignments.append(nearest)
        distances.append(nearest_distance.square())
    return torch.cat(assignments), torch.cat(distances)


def fit_motif_codebook(
    fields: list[torch.Tensor],
    *,
    spec: GridSpec,
    bundle_size: int,
    vocabulary_size: int,
    iterations: int = 20,
    max_casts: int = 100_000,
    assignment_chunk: int = 2048,
    count_weight: float = 4.0,
    seed: int = 0,
) -> tuple[MotifCodebook, dict[str, float]]:
    """Fit deterministic Lloyd prototypes over canonical local constellations."""
    if vocabulary_size <= 1 or iterations <= 0 or max_casts <= 0:
        raise ValueError("codebook hyperparameters are outside their valid range")
    normalizer = CastingNormalizer.fit(fields)
    bundles = [
        bundle_field(
            field, spec=spec, bundle_size=bundle_size, normalizer=normalizer
        )
        for field in fields
    ]
    values = torch.cat([_vectors(bundle, count_weight) for bundle in bundles])
    generator = torch.Generator(device=values.device).manual_seed(seed)
    if len(values) > max_casts:
        selected = torch.randperm(
            len(values), generator=generator, device=values.device
        )[:max_casts]
        values = values[selected]
    if len(values) < vocabulary_size:
        raise ValueError("vocabulary cannot exceed sampled casting bundles")
    initial = torch.randperm(
        len(values), generator=generator, device=values.device
    )[:vocabulary_size]
    centers = values[initial].clone()
    final_distance = None
    for _ in range(iterations):
        assignments, final_distance = _assign(
            values, centers, chunk=assignment_chunk
        )
        sums = values.new_zeros(centers.shape)
        counts = values.new_zeros(vocabulary_size)
        sums.scatter_add_(
            0, assignments[:, None].expand_as(values), values
        )
        counts.scatter_add_(0, assignments, torch.ones_like(assignments, dtype=values.dtype))
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty, None]
        if (~nonempty).any():
            replacements = torch.randperm(
                len(values), generator=generator, device=values.device
            )[: int((~nonempty).sum())]
            centers[~nonempty] = values[replacements]
    assignments, final_distance = _assign(values, centers, chunk=assignment_chunk)
    counts = torch.bincount(assignments, minlength=vocabulary_size).float()
    probabilities = counts / counts.sum()
    entropy = -(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum()
    prototype_values = centers[:, :-1].reshape(vocabulary_size, bundle_size, 22)
    codebook = MotifCodebook(
        prototypes=prototype_values,
        prototype_count_coordinates=centers[:, -1],
        normalizer=normalizer,
        grid_shape=spec.shape,
        bundle_size=bundle_size,
        count_weight=count_weight,
    )
    return codebook, {
        "sampled_casts": float(len(values)),
        "mean_squared_assignment_distance": float(final_distance.mean()),
        "utilized_fraction": float((counts > 0).float().mean()),
        "perplexity": float(entropy.exp()),
    }


def encode_program(features: torch.Tensor, codebook: MotifCodebook) -> CastProgram:
    """Assign motifs while retaining the exact continuous residual target."""
    spec = GridSpec(codebook.grid_shape, slots_per_cell=1)
    bundles = bundle_field(
        features,
        spec=spec,
        bundle_size=codebook.bundle_size,
        normalizer=codebook.normalizer,
    )
    vectors = _vectors(bundles, codebook.count_weight)
    prototype_vectors = torch.cat(
        [
            codebook.prototypes.flatten(1).to(vectors),
            codebook.prototype_count_coordinates.to(vectors)[:, None],
        ],
        dim=1,
    )
    motifs, _ = _assign(vectors, prototype_vectors, chunk=2048)
    residuals = bundles.values - codebook.prototypes.to(bundles.values)[motifs]
    return CastProgram(
        cells=bundles.cells,
        anchors=bundles.anchors,
        counts=bundles.counts,
        motifs=motifs,
        residuals=residuals,
        source_jewels=bundles.source_jewels,
    )


def decode_program(
    program: CastProgram,
    codebook: MotifCodebook,
    *,
    residual_scale: float = 1.0,
) -> torch.Tensor:
    """Cast motif constellations into continuous canonical Jewel features."""
    if not math.isfinite(residual_scale):
        raise ValueError("residual scale must be finite")
    values = codebook.prototypes.to(program.residuals)[program.motifs]
    values = values + residual_scale * program.residuals
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
        raise RuntimeError("casting decode changed the Jewel count")
    return result


def program_histogram(
    program: CastProgram, *, n_cells: int, vocabulary_size: int
) -> torch.Tensor:
    """Count cast motifs per addressed cell for gauge-tolerant comparison."""
    if n_cells <= 0 or vocabulary_size <= 0:
        raise ValueError("histogram dimensions must be positive")
    flat = program.cells * vocabulary_size + program.motifs
    return torch.bincount(
        flat, weights=program.counts.float(), minlength=n_cells * vocabulary_size
    ).reshape(n_cells, vocabulary_size)


def histogram_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    """Cosine similarity between two cell-addressed motif programs."""
    if first.shape != second.shape:
        raise ValueError("program histograms must share a shape")
    first_flat, second_flat = first.flatten().float(), second.flatten().float()
    denominator = first_flat.norm() * second_flat.norm()
    return float(first_flat.dot(second_flat) / denominator.clamp_min(1e-12))


def quantize_centers_to_cells(features: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    """Negative control replacing continuous centers with addressed cell centers."""
    result = features.clone()
    cells = spec.cell_index(features[:, :3])
    gu, gv, gt = spec.shape
    t = cells % gt
    v = (cells // gt) % gv
    u = cells // (gv * gt)
    coordinates = torch.stack((u, v, t), dim=-1).to(features)
    size = _cell_size(spec, features)
    result[:, :3] = -1.0 + (coordinates + 0.5) * size
    return result
