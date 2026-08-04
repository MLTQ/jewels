"""Validated frozen-latent dataset contract for prior training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class LatentCache:
    latents: torch.Tensor
    conditions: torch.Tensor
    names: tuple[str, ...]
    source_ids: tuple[str, ...]
    train_mask: torch.Tensor
    latent_mean: torch.Tensor
    latent_std: torch.Tensor
    condition_mean: torch.Tensor
    condition_std: torch.Tensor
    metadata: dict

    def __post_init__(self) -> None:
        samples, cells, dimensions = self.latents.shape
        if self.conditions.ndim != 2 or self.conditions.shape[0] != samples:
            raise ValueError("conditions must have shape (samples, condition_dim)")
        condition_dim = self.conditions.shape[1]
        if len(self.names) != samples or len(self.source_ids) != samples:
            raise ValueError("names and source IDs must match latent samples")
        if self.train_mask.shape != (samples,) or self.train_mask.dtype != torch.bool:
            raise ValueError("train_mask must be boolean with one value per sample")
        if self.train_mask.all() or (~self.train_mask).all():
            raise ValueError("cache must contain non-empty train and validation splits")
        if self.latent_mean.shape != (cells, dimensions):
            raise ValueError("latent_mean must have shape (cells, latent_dim)")
        if self.latent_std.shape != (cells, dimensions):
            raise ValueError("latent_std must have shape (cells, latent_dim)")
        if self.condition_mean.shape != (condition_dim,):
            raise ValueError("condition_mean must have shape (condition_dim,)")
        if self.condition_std.shape != (condition_dim,):
            raise ValueError("condition_std must have shape (condition_dim,)")
        if not torch.isfinite(self.latents).all() or not torch.isfinite(self.conditions).all():
            raise ValueError("latents and conditions must be finite")
        if not torch.isfinite(self.latent_mean).all() or (self.latent_std <= 0).any():
            raise ValueError("latent normalization must be finite and positive")
        if not torch.isfinite(self.condition_mean).all() or (self.condition_std <= 0).any():
            raise ValueError("condition normalization must be finite and positive")
        train_sources = {
            source for source, keep in zip(self.source_ids, self.train_mask, strict=True)
            if bool(keep)
        }
        validation_sources = {
            source for source, keep in zip(self.source_ids, self.train_mask, strict=True)
            if not bool(keep)
        }
        if train_sources & validation_sources:
            raise ValueError("source IDs must not cross the train/validation boundary")

    @property
    def normalized_latents(self) -> torch.Tensor:
        return (self.latents - self.latent_mean[None]) / self.latent_std[None]

    @property
    def normalized_conditions(self) -> torch.Tensor:
        return (self.conditions - self.condition_mean[None]) / self.condition_std[None]

    def split(self, train: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self.train_mask if train else ~self.train_mask
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        return self.normalized_latents[mask], self.normalized_conditions[mask], indices

    def normalize_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """Apply the image-training condition transform to image or text CLIP vectors."""
        unit = condition / condition.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return (unit - self.condition_mean.to(unit)) / self.condition_std.to(unit)

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * self.latent_std.to(normalized)[None] + self.latent_mean.to(
            normalized
        )[None]

    def state_dict(self) -> dict:
        return {
            "latents": self.latents,
            "conditions": self.conditions,
            "names": self.names,
            "source_ids": self.source_ids,
            "train_mask": self.train_mask,
            "latent_mean": self.latent_mean,
            "latent_std": self.latent_std,
            "condition_mean": self.condition_mean,
            "condition_std": self.condition_std,
            "metadata": self.metadata,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "LatentCache":
        conditions = state["conditions"].float()
        return cls(
            latents=state["latents"].float(),
            conditions=conditions,
            names=tuple(state["names"]),
            source_ids=tuple(state["source_ids"]),
            train_mask=state["train_mask"].bool(),
            latent_mean=state["latent_mean"].float(),
            latent_std=state["latent_std"].float(),
            condition_mean=state.get(
                "condition_mean", torch.zeros(conditions.shape[1])
            ).float(),
            condition_std=state.get(
                "condition_std", torch.ones(conditions.shape[1])
            ).float(),
            metadata=dict(state["metadata"]),
        )


def save_latent_cache(cache: LatentCache, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(cache.state_dict(), temporary)
    temporary.replace(destination)


def load_latent_cache(path: str | Path) -> LatentCache:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return LatentCache.from_state_dict(state)
