"""Cache frozen tokenizer latents and aligned CLIP conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sol.autoencoder import StructuredJewelAutoencoder
from sol.corpus import FeatureNormalizer, FittedExample, load_fitted_corpus
from sol.latent_data import LatentCache, save_latent_cache
from sol.prompt_embeddings import (
    PromptEmbeddingCache,
    load_prompt_cache,
    manifest_digest,
)
from sol.sparse_autoencoder import SparseJewelAutoencoder
from sol.token_grid import GridSpec


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--extra-corpus", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-cache", default="")
    parser.add_argument("--prompt-manifest", default="")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch", type=int, default=4)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _restore_tokenizer(checkpoint: dict, spec: GridSpec) -> torch.nn.Module:
    """Restore either historical padded or selected sparse tokenizer checkpoints."""
    meta = checkpoint["meta"]
    architecture = meta.get("architecture", "structured_slots_v1")
    model_class = (
        SparseJewelAutoencoder
        if architecture.startswith("sparse_variable_count")
        else StructuredJewelAutoencoder
    )
    model = model_class(**meta["model_args"], spec=spec)
    model.load_state_dict(checkpoint["model"])
    return model


def _expand_prompt_conditioned_latents(
    examples: list[FittedExample],
    latents: torch.Tensor,
    train_mask: torch.Tensor,
    manifest: dict,
    prompt_cache: PromptEmbeddingCache,
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...], tuple[str, ...], torch.Tensor]:
    """Bind fitted sources to train/evaluation prompt templates without leakage."""
    if prompt_cache.manifest_sha256 != manifest_digest(manifest):
        raise ValueError("prompt cache does not match prompt manifest")
    ownership = prompt_cache.example_prompt_indices
    manifest_examples = manifest.get("examples", [])
    if len(ownership) != len(manifest_examples):
        raise ValueError("prompt ownership does not match manifest examples")
    by_video_stem = {}
    for item, prompt_rows in zip(manifest_examples, ownership, strict=True):
        if item["source_id"] != prompt_rows["source_id"]:
            raise ValueError("prompt ownership order does not match manifest")
        stem = Path(item["video"]).stem
        if stem in by_video_stem:
            raise ValueError(f"duplicate manifest video stem: {stem}")
        by_video_stem[stem] = (item, prompt_rows)

    expanded_latents = []
    expanded_conditions = []
    expanded_names = []
    expanded_sources = []
    expanded_train = []
    for example, latent, is_train in zip(examples, latents, train_mask, strict=True):
        if example.source_id not in by_video_stem:
            raise ValueError(f"fitted source is absent from prompt manifest: {example.source_id}")
        item, prompt_rows = by_video_stem[example.source_id]
        expected_split = "train" if bool(is_train) else "validation"
        if item["split"] != expected_split or prompt_rows["split"] != expected_split:
            raise ValueError(
                f"tokenizer split disagrees with prompt manifest for {example.source_id}"
            )
        key = "train" if bool(is_train) else "evaluation"
        for prompt_index in prompt_rows[key]:
            expanded_latents.append(latent)
            expanded_conditions.append(prompt_cache.embeddings[prompt_index])
            expanded_names.append(f"{example.name}::prompt_index={prompt_index}")
            expanded_sources.append(example.source_id)
            expanded_train.append(bool(is_train))
    return (
        torch.stack(expanded_latents),
        torch.stack(expanded_conditions),
        tuple(expanded_names),
        tuple(expanded_sources),
        torch.tensor(expanded_train, dtype=torch.bool),
    )


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if args.batch <= 0:
        raise ValueError("batch must be positive")
    if bool(args.prompt_cache) != bool(args.prompt_manifest):
        raise ValueError("prompt-cache and prompt-manifest must be provided together")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = checkpoint["meta"]
    spec = GridSpec(tuple(meta["grid_shape"]), int(meta["slots_per_cell"]))
    model = _restore_tokenizer(checkpoint, spec)
    device = torch.device(args.device)
    model.to(device).eval()
    normalizer = FeatureNormalizer.from_state_dict(meta["normalizer"])
    corpus_paths = [args.corpus, *args.extra_corpus]
    examples = load_fitted_corpus(corpus_paths)
    held_out = set(meta["validation_sources"])
    train_mask = torch.tensor(
        [example.source_id not in held_out for example in examples], dtype=torch.bool
    )
    if not train_mask.any() or train_mask.all():
        raise ValueError("tokenizer validation sources do not produce a valid cache split")

    latent_batches = []
    for start in range(0, len(examples), args.batch):
        batch = examples[start : start + args.batch]
        features = torch.stack(
            [normalizer.normalize(example.features) for example in batch]
        ).to(device)
        latent_batches.append(model.encoder(features).float().cpu())
    latents = torch.cat(latent_batches)

    names = tuple(example.name for example in examples)
    source_ids = tuple(example.source_id for example in examples)
    condition_metadata = {}
    if args.prompt_cache:
        prompt_cache = load_prompt_cache(args.prompt_cache)
        manifest = json.loads(Path(args.prompt_manifest).read_text())
        latents, condition_tensor, names, source_ids, train_mask = (
            _expand_prompt_conditioned_latents(
                examples, latents, train_mask, manifest, prompt_cache
            )
        )
        condition_metadata = {
            "condition_source": "prompt_templates",
            "prompt_cache": str(Path(args.prompt_cache)),
            "prompt_cache_sha256": _sha256(Path(args.prompt_cache)),
            "prompt_manifest": str(Path(args.prompt_manifest)),
            "prompt_manifest_sha256": prompt_cache.manifest_sha256,
            "text_encoder": prompt_cache.encoder,
            "base_examples": len(examples),
            "expanded_samples": len(latents),
        }
    else:
        sidecars = {
            path.name: path.with_suffix(".clip.npy")
            for root in map(Path, corpus_paths)
            for path in root.glob("*_w*.pt")
            if not path.name.endswith(".recovery.pt")
        }
        conditions = []
        for example in examples:
            sidecar = sidecars.get(example.name)
            if sidecar is None or not sidecar.exists():
                raise FileNotFoundError(f"missing condition sidecar for {example.name}")
            condition = torch.from_numpy(np.load(sidecar)).float()
            conditions.append(condition / condition.norm().clamp_min(1e-8))
        condition_tensor = torch.stack(conditions)
        condition_metadata = {
            "condition_source": "clip_image_sidecars",
            "clip_model": "ViT-B-32/laion2b_s34b_b79k image mean-pool, renormalized",
        }
    training_latents = latents[train_mask]
    latent_mean = training_latents.mean(dim=0)
    latent_std = training_latents.std(dim=0).clamp_min(1e-3)
    training_conditions = condition_tensor[train_mask]
    condition_mean = training_conditions.mean(dim=0)
    condition_std = training_conditions.std(dim=0).clamp_min(1e-4)
    cache = LatentCache(
        latents=latents,
        conditions=condition_tensor,
        names=names,
        source_ids=source_ids,
        train_mask=train_mask,
        latent_mean=latent_mean,
        latent_std=latent_std,
        condition_mean=condition_mean,
        condition_std=condition_std,
        metadata={
            "tokenizer_checkpoint": str(checkpoint_path),
            "tokenizer_sha256": _sha256(checkpoint_path),
            "tokenizer_step": int(checkpoint["step"]),
            "tokenizer_architecture": meta.get(
                "architecture", "structured_slots_v1"
            ),
            "corpus": corpus_paths,
            "validation_sources": tuple(sorted(held_out)),
            "grid_shape": spec.shape,
            "slots_per_cell": spec.slots_per_cell,
            **condition_metadata,
        },
    )
    save_latent_cache(cache, args.out)
    normalized = cache.normalized_latents[train_mask]
    normalized_conditions = cache.normalized_conditions[train_mask]
    print(
        f"cached={len(examples)} train={int(train_mask.sum())} "
        f"validation={int((~train_mask).sum())} shape={tuple(latents.shape)} "
        f"condition_dim={condition_tensor.shape[-1]} "
        f"normalized_mean={float(normalized.mean()):.4g} "
        f"normalized_std={float(normalized.std()):.4g} "
        f"condition_mean={float(normalized_conditions.mean()):.4g} "
        f"condition_std={float(normalized_conditions.std()):.4g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
