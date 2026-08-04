"""Fit a block-PCA hierarchy and cache its coarse latent fields."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sol.block_codec import fit_block_pca
from sol.latent_data import LatentCache, load_latent_cache, save_latent_cache


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--codec-out", required=True)
    parser.add_argument("--cache-out", required=True)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--code-dim", type=int, default=96)
    parser.add_argument("--max-blocks", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cache = load_latent_cache(args.cache)
    train_latents, _, _ = cache.split(train=True)
    grid_shape = tuple(cache.metadata["grid_shape"])
    codec = fit_block_pca(
        train_latents,
        grid_shape,
        block_size=args.block_size,
        code_dim=args.code_dim,
        max_blocks=args.max_blocks,
    )
    codec_path = Path(args.codec_out)
    codec_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = codec_path.with_suffix(codec_path.suffix + ".tmp")
    torch.save(codec.state_dict(), temporary)
    temporary.replace(codec_path)

    normalized = cache.normalized_latents
    codes = codec.encode(normalized)
    training_codes = codes[cache.train_mask]
    coarse_cache = LatentCache(
        latents=codes,
        conditions=cache.conditions,
        names=cache.names,
        source_ids=cache.source_ids,
        train_mask=cache.train_mask,
        latent_mean=training_codes.mean(dim=0),
        latent_std=training_codes.std(dim=0).clamp_min(1e-3),
        condition_mean=cache.condition_mean,
        condition_std=cache.condition_std,
        metadata={
            **cache.metadata,
            "parent_cache": str(args.cache),
            "block_codec": str(codec_path),
            "fine_grid_shape": grid_shape,
            "grid_shape": codec.coarse_shape,
            "block_size": codec.block_size,
            "block_code_dim": codec.code_dim,
            "block_explained_variance": codec.explained_variance,
        },
    )
    save_latent_cache(coarse_cache, args.cache_out)
    print(
        f"fine={tuple(normalized.shape)} coarse={tuple(codes.shape)} "
        f"compression={normalized.numel() / codes.numel():.3f}x "
        f"explained={codec.explained_variance:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
