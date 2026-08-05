"""Validated text-embedding sidecar contract for prompted jewel corpora."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from sol.ucf_prompt_manifest import SCHEMA as MANIFEST_SCHEMA


SCHEMA = "jewel-prompt-embeddings-v1"


def manifest_digest(manifest: dict) -> str:
    """Hash the canonical manifest so embeddings cannot attach to changed prompts."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def collect_prompts(manifest: dict) -> tuple[str, ...]:
    """Collect unique train/evaluation prompts in stable manifest order."""
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported prompt manifest schema")
    prompts = []
    seen = set()
    for example in manifest.get("examples", []):
        train = tuple(example.get("train_prompts", ()))
        evaluation = tuple(example.get("evaluation_prompts", ()))
        if not train or not evaluation or set(train) & set(evaluation):
            raise ValueError("each example needs disjoint train and evaluation prompts")
        for prompt in train + evaluation:
            normalized = " ".join(prompt.split())
            if not normalized:
                raise ValueError("prompt text cannot be empty")
            if normalized not in seen:
                seen.add(normalized)
                prompts.append(normalized)
    if not prompts:
        raise ValueError("manifest contains no prompts")
    return tuple(prompts)


@dataclass(frozen=True)
class PromptEmbeddingCache:
    prompts: tuple[str, ...]
    embeddings: torch.Tensor
    example_prompt_indices: tuple[dict, ...]
    encoder: dict
    manifest_sha256: str

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.prompts):
            raise ValueError("embeddings must have one row per prompt")
        if not self.prompts or len(self.prompts) != len(set(self.prompts)):
            raise ValueError("prompts must be non-empty and unique")
        if not torch.isfinite(self.embeddings).all():
            raise ValueError("prompt embeddings must be finite")
        norms = self.embeddings.float().norm(dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=2e-4, rtol=2e-4):
            raise ValueError("prompt embeddings must be unit normalized")
        if len(self.manifest_sha256) != 64:
            raise ValueError("manifest_sha256 must be a SHA-256 hex digest")
        for example in self.example_prompt_indices:
            for key in ("train", "evaluation"):
                indices = example.get(key)
                if not indices or min(indices) < 0 or max(indices) >= len(self.prompts):
                    raise ValueError("example prompt indices are missing or out of range")

    def state_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "prompts": self.prompts,
            "embeddings": self.embeddings,
            "example_prompt_indices": self.example_prompt_indices,
            "encoder": self.encoder,
            "manifest_sha256": self.manifest_sha256,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "PromptEmbeddingCache":
        if state.get("schema") != SCHEMA:
            raise ValueError("unsupported prompt embedding schema")
        return cls(
            prompts=tuple(state["prompts"]),
            embeddings=state["embeddings"].float(),
            example_prompt_indices=tuple(state["example_prompt_indices"]),
            encoder=dict(state["encoder"]),
            manifest_sha256=state["manifest_sha256"],
        )


def build_prompt_cache(
    manifest: dict,
    prompts: tuple[str, ...],
    embeddings: torch.Tensor,
) -> PromptEmbeddingCache:
    """Bind encoded prompt rows back to each manifest example."""
    expected = collect_prompts(manifest)
    if prompts != expected:
        raise ValueError("encoded prompt order does not match the manifest")
    lookup = {prompt: index for index, prompt in enumerate(prompts)}
    ownership = []
    for example in manifest["examples"]:
        ownership.append(
            {
                "source_id": example["source_id"],
                "class_id": example["class_id"],
                "split": example["split"],
                "train": [lookup[" ".join(text.split())] for text in example["train_prompts"]],
                "evaluation": [
                    lookup[" ".join(text.split())]
                    for text in example["evaluation_prompts"]
                ],
            }
        )
    return PromptEmbeddingCache(
        prompts=prompts,
        embeddings=embeddings.float(),
        example_prompt_indices=tuple(ownership),
        encoder=dict(manifest["text_encoder"]),
        manifest_sha256=manifest_digest(manifest),
    )


def save_prompt_cache(cache: PromptEmbeddingCache, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(cache.state_dict(), temporary)
    temporary.replace(destination)


def load_prompt_cache(path: str | Path) -> PromptEmbeddingCache:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return PromptEmbeddingCache.from_state_dict(state)
