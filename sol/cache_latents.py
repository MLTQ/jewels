"""Cache frozen tokenizer latents and aligned CLIP conditions."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from sol.autoencoder import StructuredJewelAutoencoder
from sol.corpus import FeatureNormalizer, load_fitted_corpus
from sol.latent_data import LatentCache, save_latent_cache
from sol.sparse_autoencoder import SparseJewelAutoencoder
from sol.token_grid import GridSpec


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
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


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if args.batch <= 0:
        raise ValueError("batch must be positive")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = checkpoint["meta"]
    spec = GridSpec(tuple(meta["grid_shape"]), int(meta["slots_per_cell"]))
    model = _restore_tokenizer(checkpoint, spec)
    device = torch.device(args.device)
    model.to(device).eval()
    normalizer = FeatureNormalizer.from_state_dict(meta["normalizer"])
    examples = load_fitted_corpus(args.corpus)
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

    corpus_path = Path(args.corpus)
    conditions = []
    for example in examples:
        sidecar = corpus_path / Path(example.name).with_suffix(".clip.npy")
        if not sidecar.exists():
            raise FileNotFoundError(f"missing condition sidecar {sidecar}")
        condition = torch.from_numpy(np.load(sidecar)).float()
        conditions.append(condition / condition.norm().clamp_min(1e-8))
    condition_tensor = torch.stack(conditions)
    training_latents = latents[train_mask]
    latent_mean = training_latents.mean(dim=0)
    latent_std = training_latents.std(dim=0).clamp_min(1e-3)
    training_conditions = condition_tensor[train_mask]
    condition_mean = training_conditions.mean(dim=0)
    condition_std = training_conditions.std(dim=0).clamp_min(1e-4)
    cache = LatentCache(
        latents=latents,
        conditions=condition_tensor,
        names=tuple(example.name for example in examples),
        source_ids=tuple(example.source_id for example in examples),
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
            "corpus": str(args.corpus),
            "validation_sources": tuple(sorted(held_out)),
            "grid_shape": spec.shape,
            "slots_per_cell": spec.slots_per_cell,
            "clip_model": "ViT-B-32/laion2b_s34b_b79k image mean-pool, renormalized",
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
