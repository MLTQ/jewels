"""Learned product-code decoder for hierarchical Jewel casting phrases."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from sol.factorized_jewel_casting_language import (
    FactorCodebook,
    FactorizedCodebook,
    FactorizedProgram,
    _assign,
    composed_prototypes,
    encode_factorized_program,
)
from sol.jewel_casting_language import _cell_size, bundle_field
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class HierarchicalPhraseBatch:
    """Aligned pair geometry and individual appearance phrase targets."""

    tokens: torch.Tensor
    cells: torch.Tensor
    anchors: torch.Tensor
    counts: torch.Tensor
    base_values: torch.Tensor
    target_values: torch.Tensor

    def to(self, device: torch.device | str) -> "HierarchicalPhraseBatch":
        return HierarchicalPhraseBatch(
            tokens=self.tokens.to(device),
            cells=self.cells.to(device),
            anchors=self.anchors.to(device),
            counts=self.counts.to(device),
            base_values=self.base_values.to(device),
            target_values=self.target_values.to(device),
        )

    def index(self, selected: torch.Tensor) -> "HierarchicalPhraseBatch":
        return HierarchicalPhraseBatch(
            tokens=self.tokens[selected],
            cells=self.cells[selected],
            anchors=self.anchors[selected],
            counts=self.counts[selected],
            base_values=self.base_values[selected],
            target_values=self.target_values[selected],
        )

    def __len__(self) -> int:
        return int(len(self.cells))


def build_hierarchical_phrase_batch(
    features: torch.Tensor,
    pair_codebook: FactorizedCodebook,
    individual_codebook: FactorizedCodebook,
) -> tuple[HierarchicalPhraseBatch, FactorizedProgram, FactorizedProgram]:
    """Align active pair and individual role tokens with exact pair-bundle targets."""
    if pair_codebook.bundle_size != 2 or individual_codebook.bundle_size != 1:
        raise ValueError("hierarchical decoder requires bundle sizes two and one")
    if pair_codebook.grid_shape != individual_codebook.grid_shape:
        raise ValueError("hierarchical codebooks must share the address grid")
    pair = encode_factorized_program(features, pair_codebook)
    individual = encode_factorized_program(features, individual_codebook)
    if individual.casts != pair.source_jewels:
        raise RuntimeError("individual program must emit one cast per Jewel")
    owner = torch.repeat_interleave(
        torch.arange(pair.casts, device=features.device), pair.counts
    )
    starts = torch.cumsum(pair.counts, dim=0) - pair.counts
    within = torch.arange(individual.casts, device=features.device) - torch.repeat_interleave(
        starts, pair.counts
    )
    if not torch.equal(pair.cells[owner], individual.cells):
        raise RuntimeError("pair and individual programs disagree on canonical cell order")
    vocabulary_size = pair_codebook.vocabulary_size
    tokens = torch.full(
        (pair.casts, 6), vocabulary_size, device=features.device, dtype=torch.long
    )
    tokens[:, 0] = pair.tokens["layout"]
    tokens[:, 1] = pair.tokens["covariance"]
    tokens[owner, 2 + within] = individual.tokens["surface"]
    tokens[owner, 4 + within] = individual.tokens["gradient"]

    pair_values = composed_prototypes(pair, pair_codebook)
    individual_values = composed_prototypes(individual, individual_codebook)[:, 0]
    base_values = pair_values.clone()
    base_values[:, :, 9:] = 0
    base_values[owner, within, 9:] = individual_values[:, 9:]
    target_values = pair_values + pair.residuals
    return (
        HierarchicalPhraseBatch(
            tokens=tokens,
            cells=pair.cells,
            anchors=pair.anchors,
            counts=pair.counts,
            base_values=base_values,
            target_values=target_values,
        ),
        pair,
        individual,
    )


def _factor(codebook: FactorizedCodebook, name: str) -> FactorCodebook:
    return next(factor for factor in codebook.factors if factor.name == name)


def _assign_pair_factor(
    values: torch.Tensor,
    counts: torch.Tensor,
    factor: FactorCodebook,
    count_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = values[:, :, list(factor.dimensions)]
    vectors = torch.cat(
        [
            selected.flatten(1),
            counts.to(values)[:, None] / values.shape[1] * count_weight,
        ],
        dim=1,
    )
    prototypes = torch.cat(
        [
            factor.prototypes.flatten(1).to(values),
            factor.prototype_count_coordinates.to(values)[:, None],
        ],
        dim=1,
    )
    tokens, _ = _assign(vectors, prototypes, chunk=2048)
    return tokens, factor.prototypes.to(values)[tokens]


def _assign_individual_factor(
    values: torch.Tensor,
    factor: FactorCodebook,
    count_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = values[:, list(factor.dimensions)]
    vectors = torch.cat(
        [selected, selected.new_full((len(selected), 1), count_weight)], dim=1
    )
    prototypes = torch.cat(
        [
            factor.prototypes[:, 0].to(values),
            factor.prototype_count_coordinates.to(values)[:, None],
        ],
        dim=1,
    )
    tokens, _ = _assign(vectors, prototypes, chunk=2048)
    return tokens, factor.prototypes[:, 0].to(values)[tokens]


def build_sampled_hierarchical_phrase_batch(
    features: torch.Tensor,
    pair_codebook: FactorizedCodebook,
    individual_codebook: FactorizedCodebook,
    *,
    max_pairs: int,
    generator: torch.Generator,
) -> HierarchicalPhraseBatch:
    """Build a balanced training sample without assigning unused full-field roles."""
    if max_pairs <= 0:
        raise ValueError("sampled phrase batches require a positive pair budget")
    spec = GridSpec(pair_codebook.grid_shape, slots_per_cell=1)
    bundles = bundle_field(
        features,
        spec=spec,
        bundle_size=2,
        normalizer=pair_codebook.normalizer,
    )
    selected = torch.randperm(
        bundles.casts, generator=generator, device=features.device
    )[: min(max_pairs, bundles.casts)]
    values = bundles.values[selected]
    counts = bundles.counts[selected]
    pair_layout, layout_values = _assign_pair_factor(
        values, counts, _factor(pair_codebook, "layout"), pair_codebook.count_weight
    )
    pair_covariance, covariance_values = _assign_pair_factor(
        values,
        counts,
        _factor(pair_codebook, "covariance"),
        pair_codebook.count_weight,
    )
    owner = torch.repeat_interleave(
        torch.arange(len(selected), device=features.device), counts
    )
    starts = torch.cumsum(counts, dim=0) - counts
    within = torch.arange(int(counts.sum()), device=features.device) - torch.repeat_interleave(
        starts, counts
    )
    individual_values = values[owner, within]
    surface, surface_values = _assign_individual_factor(
        individual_values,
        _factor(individual_codebook, "surface"),
        individual_codebook.count_weight,
    )
    gradient, gradient_values = _assign_individual_factor(
        individual_values,
        _factor(individual_codebook, "gradient"),
        individual_codebook.count_weight,
    )
    padding = pair_codebook.vocabulary_size
    tokens = torch.full(
        (len(selected), 6), padding, device=features.device, dtype=torch.long
    )
    tokens[:, 0], tokens[:, 1] = pair_layout, pair_covariance
    tokens[owner, 2 + within], tokens[owner, 4 + within] = surface, gradient
    base_values = values.new_zeros(values.shape)
    base_values[:, :, :3] = layout_values
    base_values[:, :, 3:9] = covariance_values
    base_values[owner, within, 9:12] = surface_values[:, :3]
    base_values[owner, within, 21] = surface_values[:, 3]
    base_values[owner, within, 12:21] = gradient_values
    return HierarchicalPhraseBatch(
        tokens=tokens,
        cells=bundles.cells[selected],
        anchors=bundles.anchors[selected],
        counts=counts,
        base_values=base_values,
        target_values=values,
    )


def concatenate_phrase_batches(
    batches: list[HierarchicalPhraseBatch],
) -> HierarchicalPhraseBatch:
    """Concatenate source-owned phrase rows without losing field boundaries elsewhere."""
    if not batches:
        raise ValueError("cannot concatenate an empty phrase dataset")
    return HierarchicalPhraseBatch(
        tokens=torch.cat([batch.tokens for batch in batches]),
        cells=torch.cat([batch.cells for batch in batches]),
        anchors=torch.cat([batch.anchors for batch in batches]),
        counts=torch.cat([batch.counts for batch in batches]),
        base_values=torch.cat([batch.base_values for batch in batches]),
        target_values=torch.cat([batch.target_values for batch in batches]),
    )


def residual_scale(batch: HierarchicalPhraseBatch) -> torch.Tensor:
    """Compute train-owned per-row/per-feature RMS scales for decoder targets."""
    residual = batch.target_values - batch.base_values
    row = torch.arange(2, device=batch.counts.device)
    valid = row[None] < batch.counts[:, None]
    scale = residual.new_ones(2, 22)
    for row_index in range(2):
        selected = residual[valid[:, row_index], row_index]
        if len(selected):
            scale[row_index] = selected.square().mean(dim=0).sqrt().clamp_min(1e-4)
    return scale


class HierarchicalPhraseDecoder(nn.Module):
    """Decode product tokens and an irregular anchor into continuous Jewel corrections."""

    def __init__(
        self,
        *,
        vocabulary_size: int = 1024,
        n_cells: int = 256,
        embedding_dim: int = 48,
        hidden_dim: int = 512,
        depth: int = 4,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if min(vocabulary_size, n_cells, embedding_dim, hidden_dim, depth) <= 0:
            raise ValueError("decoder dimensions must be positive")
        self.vocabulary_size = vocabulary_size
        self.n_cells = n_cells
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.role_embeddings = nn.ModuleList(
            [
                nn.Embedding(vocabulary_size + 1, embedding_dim, padding_idx=vocabulary_size)
                for _ in range(6)
            ]
        )
        self.cell_embedding = nn.Embedding(n_cells, embedding_dim)
        self.count_embedding = nn.Embedding(3, embedding_dim // 2)
        anchor_dim = 3 * (1 + 2 * 3)
        input_dim = embedding_dim * 7 + embedding_dim // 2 + anchor_dim
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 44))
        self.network = nn.Sequential(*layers)
        scale = torch.ones(2, 22) if output_scale is None else output_scale.float()
        if scale.shape != (2, 22):
            raise ValueError("decoder output scale must have shape (2,22)")
        self.register_buffer("output_scale", scale)

    def _anchor_features(self, anchors: torch.Tensor) -> torch.Tensor:
        features = [anchors]
        for frequency in (1.0, 2.0, 4.0):
            features.extend(
                [
                    torch.sin(torch.pi * frequency * anchors),
                    torch.cos(torch.pi * frequency * anchors),
                ]
            )
        return torch.cat(features, dim=1)

    def forward(self, batch: HierarchicalPhraseBatch) -> torch.Tensor:
        role = [
            embedding(batch.tokens[:, index])
            for index, embedding in enumerate(self.role_embeddings)
        ]
        inputs = torch.cat(
            role
            + [
                self.cell_embedding(batch.cells),
                self.count_embedding(batch.counts),
                self._anchor_features(batch.anchors),
            ],
            dim=1,
        )
        normalized = self.network(inputs).reshape(-1, 2, 22)
        correction = 3.0 * torch.tanh(normalized / 3.0) * self.output_scale
        return batch.base_values + correction

    def architecture(self) -> dict:
        return {
            "vocabulary_size": self.vocabulary_size,
            "n_cells": self.n_cells,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
        }


def phrase_decoder_loss(
    prediction: torch.Tensor,
    batch: HierarchicalPhraseBatch,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Masked train-normalized residual error over active Jewel rows."""
    if prediction.shape != batch.target_values.shape or scale.shape != (2, 22):
        raise ValueError("decoder loss inputs have incompatible shapes")
    normalized = (prediction - batch.target_values) / scale.clamp_min(1e-4)
    row = torch.arange(2, device=batch.counts.device)
    valid = row[None] < batch.counts[:, None]
    return normalized[valid].square().mean()


def phrase_values_to_features(
    values: torch.Tensor,
    pair_program: FactorizedProgram,
    pair_codebook: FactorizedCodebook,
) -> torch.Tensor:
    """Decode predicted normalized pair values through the canonical feature contract."""
    if values.shape != (pair_program.casts, 2, 22):
        raise ValueError("predicted pair values have an incompatible shape")
    spec = GridSpec(pair_codebook.grid_shape, slots_per_cell=1)
    size = _cell_size(spec, values)
    mean = pair_codebook.normalizer.intrinsic_mean.to(values)
    std = pair_codebook.normalizer.intrinsic_std.to(values)
    output = []
    for index, count in enumerate(pair_program.counts.tolist()):
        part = values[index, :count]
        decoded = values.new_empty(count, 22)
        decoded[:, :3] = pair_program.anchors[index] + part[:, :3] * size
        decoded[:, 3:] = part[:, 3:] * std + mean
        output.append(decoded)
    return torch.cat(output)
