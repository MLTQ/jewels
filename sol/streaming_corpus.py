"""Multi-clip prompted continuation corpus with train-only shared normalization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Sequence

import torch

from sol.corpus import _state_to_features
from sol.prompt_embeddings import PromptEmbeddingCache, manifest_digest
from sol.streaming_data import (
    ContinuationDataset,
    FeatureStandardizer,
    build_continuation_dataset,
)
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class PromptedField:
    source_id: str
    class_id: int
    class_name: str
    split: str
    features: torch.Tensor
    frames: int
    train_prompt_indices: tuple[int, ...]
    evaluation_prompt_indices: tuple[int, ...]


@dataclass(frozen=True)
class PromptedContinuationExample:
    source_id: str
    class_id: int
    class_name: str
    split: str
    dataset: ContinuationDataset
    train_prompt_indices: tuple[int, ...]
    evaluation_prompt_indices: tuple[int, ...]


@dataclass(frozen=True)
class PromptedContinuationCorpus:
    examples: tuple[PromptedContinuationExample, ...]
    prompt_embeddings: torch.Tensor
    context_standardizer: FeatureStandardizer
    birth_standardizer: FeatureStandardizer

    @property
    def train(self) -> tuple[PromptedContinuationExample, ...]:
        return tuple(example for example in self.examples if example.split == "train")

    @property
    def validation(self) -> tuple[PromptedContinuationExample, ...]:
        return tuple(example for example in self.examples if example.split == "validation")


def load_prompted_fields(
    manifest: dict,
    prompt_cache: PromptEmbeddingCache,
    checkpoint_roots: str | Path | Sequence[str | Path],
) -> list[PromptedField]:
    """Load one fitted 96-frame field for every manifest example."""
    if prompt_cache.manifest_sha256 != manifest_digest(manifest):
        raise ValueError("prompt cache does not match the fitted manifest")
    roots = (
        [Path(root) for root in checkpoint_roots]
        if isinstance(checkpoint_roots, Sequence)
        and not isinstance(checkpoint_roots, (str, bytes))
        else [Path(checkpoint_roots)]
    )
    checkpoints = {}
    for root in roots:
        for path in root.glob("*_w000000.pt"):
            if path.name in checkpoints:
                raise ValueError(f"duplicate fitted checkpoint: {path.name}")
            checkpoints[path.name] = path
    ownership = {
        item["source_id"]: item for item in prompt_cache.example_prompt_indices
    }
    fields = []
    missing = []
    for example in manifest["examples"]:
        name = f"{Path(example['video']).stem}_w000000.pt"
        path = checkpoints.get(name)
        if path is None:
            missing.append(name)
            continue
        saved = torch.load(path, map_location="cpu", weights_only=False)
        shape = tuple(saved["info"]["shape"])
        if shape[0] != example["frames"]:
            raise ValueError(f"fitted frame count disagrees with manifest: {path}")
        prompt_owner = ownership[example["source_id"]]
        if prompt_owner["split"] != example["split"]:
            raise ValueError("prompt ownership split disagrees with manifest")
        fields.append(
            PromptedField(
                source_id=example["source_id"],
                class_id=example["class_id"],
                class_name=example["class_name"],
                split=example["split"],
                features=_state_to_features(saved["state"]).float(),
                frames=shape[0],
                train_prompt_indices=tuple(prompt_owner["train"]),
                evaluation_prompt_indices=tuple(prompt_owner["evaluation"]),
            )
        )
    if missing:
        raise FileNotFoundError(f"missing fitted prompt fields: {sorted(missing)}")
    return fields


def build_prompted_continuation_corpus(
    fields: list[PromptedField],
    prompt_embeddings: torch.Tensor,
    *,
    prefix_frames: int = 32,
    stride_frames: int = 16,
    support_sigma: float = 3.0,
    grid_spec: GridSpec = GridSpec((16, 16, 8), 256),
) -> PromptedContinuationCorpus:
    """Build all views, then replace per-clip statistics with train-only shared statistics."""
    if not fields:
        raise ValueError("prompted continuation corpus requires fitted fields")
    if prompt_embeddings.ndim != 2 or not torch.isfinite(prompt_embeddings).all():
        raise ValueError("prompt embeddings must be a finite matrix")
    source_ids = [field.source_id for field in fields]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("prompted fields require unique source IDs")
    raw = []
    for field in fields:
        if field.split not in {"train", "validation"}:
            raise ValueError("field split must be train or validation")
        for index in field.train_prompt_indices + field.evaluation_prompt_indices:
            if not 0 <= index < len(prompt_embeddings):
                raise ValueError("field prompt index is out of range")
        raw.append(
            build_continuation_dataset(
                field.features,
                field.frames,
                prefix_frames=prefix_frames,
                stride_frames=stride_frames,
                support_sigma=support_sigma,
                grid_spec=grid_spec,
            )
        )
    train_indices = [index for index, field in enumerate(fields) if field.split == "train"]
    validation_indices = [
        index for index, field in enumerate(fields) if field.split == "validation"
    ]
    if not train_indices or not validation_indices:
        raise ValueError("corpus requires non-empty train and validation fields")
    train_classes = {fields[index].class_id for index in train_indices}
    validation_classes = {fields[index].class_id for index in validation_indices}
    if train_classes != validation_classes:
        raise ValueError("every prompt class must occur in both corpus splits")
    context_standardizer = FeatureStandardizer.fit(
        [
            view.context_features
            for index in train_indices
            for view in raw[index].views
        ]
    )
    birth_standardizer = FeatureStandardizer.fit(
        [view.births.values for index in train_indices for view in raw[index].views]
    )
    examples = []
    for field, dataset in zip(fields, raw, strict=True):
        shared_dataset = replace(
            dataset,
            context_standardizer=context_standardizer,
            birth_standardizer=birth_standardizer,
        )
        examples.append(
            PromptedContinuationExample(
                source_id=field.source_id,
                class_id=field.class_id,
                class_name=field.class_name,
                split=field.split,
                dataset=shared_dataset,
                train_prompt_indices=field.train_prompt_indices,
                evaluation_prompt_indices=field.evaluation_prompt_indices,
            )
        )
    return PromptedContinuationCorpus(
        examples=tuple(examples),
        prompt_embeddings=prompt_embeddings.float(),
        context_standardizer=context_standardizer,
        birth_standardizer=birth_standardizer,
    )
