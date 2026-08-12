"""Initial-plus-continuation mark corpus and generated-state window selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.scaffold_topology_data import ScaffoldTopologyView, build_scaffold_topology_views
from sol.streaming import build_rolling_windows, measure_jewel_lifecycles
from sol.streaming_corpus import PromptedField
from sol.streaming_data import FeatureStandardizer, rasterize_context
from sol.streaming_features import to_frontier_time
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class ScaffoldMarkSource:
    """One prompted fitted field and all of its complete emission strides."""

    field: PromptedField
    views: tuple[ScaffoldTopologyView, ...]


@dataclass(frozen=True)
class ScaffoldMarkCorpus:
    """Leakage-safe mark corpus sharing train-only feature statistics."""

    sources: tuple[ScaffoldMarkSource, ...]
    prompt_embeddings: torch.Tensor
    context_standardizer: FeatureStandardizer
    birth_standardizer: FeatureStandardizer
    grid_spec: GridSpec
    stride_frames: int
    support_sigma: float

    @property
    def train(self) -> tuple[ScaffoldMarkSource, ...]:
        return tuple(source for source in self.sources if source.field.split == "train")

    @property
    def validation(self) -> tuple[ScaffoldMarkSource, ...]:
        return tuple(
            source for source in self.sources if source.field.split == "validation"
        )


@dataclass(frozen=True)
class GeneratedWindowState:
    """Causal context/carry row selections from the currently generated field."""

    context_features: torch.Tensor
    context_row_indices: torch.Tensor
    carried_global_features: torch.Tensor
    carried_row_indices: torch.Tensor
    active_commit_row_indices: torch.Tensor


def build_scaffold_mark_corpus(
    fields: list[PromptedField],
    prompt_embeddings: torch.Tensor,
    *,
    stride_frames: int = 16,
    support_sigma: float = 3.0,
    grid_spec: GridSpec = GridSpec((16, 16, 8), 1024),
) -> ScaffoldMarkCorpus:
    """Build all full strides and fit normalization on training sources only."""
    if not fields:
        raise ValueError("scaffold mark corpus requires fitted fields")
    if prompt_embeddings.ndim != 2 or not torch.isfinite(prompt_embeddings).all():
        raise ValueError("prompt embeddings must be a finite matrix")
    source_ids = [field.source_id for field in fields]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("scaffold mark fields require unique source IDs")
    sources = []
    for field in fields:
        if field.split not in {"train", "validation"}:
            raise ValueError("field split must be train or validation")
        for index in field.train_prompt_indices + field.evaluation_prompt_indices:
            if not 0 <= index < len(prompt_embeddings):
                raise ValueError("field prompt index is out of range")
        sources.append(
            ScaffoldMarkSource(
                field,
                build_scaffold_topology_views(
                    field.features,
                    field.frames,
                    stride_frames=stride_frames,
                    support_sigma=support_sigma,
                    grid_spec=grid_spec,
                ),
            )
        )
    train = [source for source in sources if source.field.split == "train"]
    validation = [
        source for source in sources if source.field.split == "validation"
    ]
    if not train or not validation:
        raise ValueError("scaffold mark corpus requires train and validation sources")
    if {source.field.class_id for source in train} != {
        source.field.class_id for source in validation
    }:
        raise ValueError("every prompt class must occur in both corpus splits")
    context_standardizer = FeatureStandardizer.fit(
        [view.context_features for source in train for view in source.views]
    )
    birth_standardizer = FeatureStandardizer.fit(
        [view.births.values for source in train for view in source.views]
    )
    return ScaffoldMarkCorpus(
        tuple(sources),
        prompt_embeddings.float(),
        context_standardizer,
        birth_standardizer,
        grid_spec,
        stride_frames,
        support_sigma,
    )


def rasterize_scaffold_context(
    features: torch.Tensor,
    standardizer: FeatureStandardizer,
    *,
    stride_frames: int,
    grid_spec: GridSpec,
) -> torch.Tensor:
    """Rasterize a one-stride context, including the empty initial condition."""
    if features.ndim != 2 or features.shape[1] != 22:
        raise ValueError("context features must have shape (jewels,22)")
    if not len(features):
        return features.new_zeros(grid_spec.n_cells, 46)
    return rasterize_context(
        features,
        standardizer,
        prefix_frames=stride_frames,
        stride_frames=stride_frames,
        grid_shape=grid_spec.shape,
    )


def generated_window_state(
    features: torch.Tensor,
    total_frames: int,
    frontier: int,
    *,
    stride_frames: int,
    support_sigma: float,
) -> GeneratedWindowState:
    """Select causal context and carry from an append-only generated field."""
    if features.ndim != 2 or features.shape[1] != 22:
        raise ValueError("generated features must have shape (jewels,22)")
    if frontier < 0 or frontier % stride_frames or frontier + stride_frames > total_frames:
        raise ValueError("frontier must begin one complete aligned stride")
    if not len(features):
        empty = torch.empty(0, dtype=torch.long, device=features.device)
        return GeneratedWindowState(features.clone(), empty, features.clone(), empty, empty)
    lifecycles = measure_jewel_lifecycles(
        features, total_frames, support_sigma=support_sigma
    )
    windows = build_rolling_windows(
        lifecycles,
        total_frames,
        prefix_frames=stride_frames,
        stride_frames=stride_frames,
    )
    window = next((item for item in windows if item.frontier == frontier), None)
    if window is None or window.commit_stop - window.frontier != stride_frames:
        raise ValueError("frontier does not own a complete generated-state window")
    return GeneratedWindowState(
        context_features=to_frontier_time(
            features[window.context_ids], total_frames, frontier, stride_frames
        ),
        context_row_indices=window.context_ids,
        carried_global_features=features[window.carried_ids].clone(),
        carried_row_indices=window.carried_ids,
        active_commit_row_indices=window.active_commit_ids,
    )
