"""Adapter from fitted stprim checkpoints to leakage-safe sol/ training examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys
from collections.abc import Sequence

import torch


@dataclass
class FittedExample:
    name: str
    source_id: str
    features: torch.Tensor
    background: torch.Tensor
    shape: tuple[int, int, int]
    domain_id: str = "default"


@dataclass
class SourceSplit:
    train: list[FittedExample]
    validation: list[FittedExample]
    validation_sources: tuple[str, ...]


@dataclass
class FeatureNormalizer:
    """Standardize appearance/geometry while retaining physical center coordinates."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(
        cls,
        examples: list[FittedExample],
        *,
        balance_domains: bool = False,
    ) -> "FeatureNormalizer":
        if not examples:
            raise ValueError("cannot fit normalization without examples")
        groups: dict[str, list[FittedExample]] = {}
        for example in examples:
            key = example.domain_id if balance_domains else "all"
            groups.setdefault(key, []).append(example)
        means = []
        second_moments = []
        for group in groups.values():
            values = torch.cat([example.features.double() for example in group])
            means.append(values.mean(dim=0))
            second_moments.append(values.square().mean(dim=0))
        mean = torch.stack(means).mean(dim=0)
        second_moment = torch.stack(second_moments).mean(dim=0)
        variance = (second_moment - mean.square()).clamp_min(0)
        std = variance.sqrt().clamp_min(1e-4)
        mean[:3] = 0
        std[:3] = 1
        return cls(mean.float(), std.float())

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean.to(features)) / self.std.to(features)

    def denormalize(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.std.to(features) + self.mean.to(features)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "FeatureNormalizer":
        return cls(state["mean"].float(), state["std"].float())


def _state_to_features(state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Use the production featurizer without changing stprim package layout."""
    stprim_root = Path(__file__).resolve().parent.parent / "stprim"
    path = str(stprim_root)
    if path not in sys.path:
        sys.path.insert(0, path)
    from prior.featurize import state_to_features

    return state_to_features(state)


def load_fitted_corpus(
    corpus_dir: str | Path | Sequence[str | Path],
    *,
    limit: int = 0,
) -> list[FittedExample]:
    """Load fitted checkpoints but not CLIP sidecars into CPU feature tensors."""
    roots = (
        [Path(path) for path in corpus_dir]
        if isinstance(corpus_dir, Sequence) and not isinstance(corpus_dir, (str, bytes))
        else [Path(corpus_dir)]
    )
    paths = sorted(
        (path, root.name)
        for root in roots
        for path in root.glob("*_w*.pt")
        if not path.name.endswith((".recovery.pt", ".motion.pt"))
    )
    if limit:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no fitted checkpoints in {roots}")
    names = [path.name for path, _ in paths]
    if len(names) != len(set(names)):
        raise ValueError("corpus roots contain duplicate checkpoint filenames")
    examples = []
    for path, domain_id in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        source = data.get("source", {})
        video = source.get("video")
        source_id = Path(video).stem if video else path.stem.split("_w", 1)[0]
        examples.append(
            FittedExample(
                name=path.name,
                source_id=source_id,
                features=_state_to_features(data["state"]).float(),
                background=torch.tensor(data["info"]["background"]).float(),
                shape=tuple(data["info"]["shape"]),
                domain_id=domain_id,
            )
        )
    counts = {example.features.shape[0] for example in examples}
    if len(counts) != 1:
        raise ValueError(f"corpus contains multiple jewel counts: {sorted(counts)}")
    return examples


def split_by_source(
    examples: list[FittedExample],
    *,
    validation_sources: int = 2,
    seed: int = 0,
    held_out_sources: Sequence[str] | None = None,
) -> SourceSplit:
    """Hold out whole source videos so adjacent windows cannot leak across splits."""
    sources = sorted({example.source_id for example in examples})
    if held_out_sources is None:
        if validation_sources <= 0 or validation_sources >= len(sources):
            raise ValueError("validation_sources must leave at least one source for each split")
        random.Random(seed).shuffle(sources)
        held_out = tuple(sorted(sources[:validation_sources]))
    else:
        held_out = tuple(sorted(set(held_out_sources)))
        missing = set(held_out) - set(sources)
        if missing:
            raise ValueError(f"held-out sources are absent from corpus: {sorted(missing)}")
        if not held_out or len(held_out) == len(sources):
            raise ValueError("held-out sources must leave at least one source for each split")
    held_out_set = set(held_out)
    train = [example for example in examples if example.source_id not in held_out_set]
    validation = [example for example in examples if example.source_id in held_out_set]
    if not train or not validation:
        raise RuntimeError("source split unexpectedly produced an empty partition")
    return SourceSplit(train, validation, held_out)
