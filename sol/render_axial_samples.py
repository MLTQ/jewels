"""Render fitted targets, block roundtrips, and axial-prior samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol.block_codec import BlockPCACodec
from sol.cache_latents import _restore_tokenizer
from sol.corpus import FeatureNormalizer, load_fitted_corpus
from sol.evaluate_latent_prior import _restore_prior
from sol.evaluation import select_balanced_examples
from sol.latent_data import load_latent_cache
from sol.latent_prior import sample_flow
from sol.render_prior_samples import _panel, _render, _row
from sol.token_grid import GridSpec


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", required=True)
    parser.add_argument("--coarse-cache", required=True)
    parser.add_argument("--fine-cache", required=True)
    parser.add_argument("--block-codec", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--knn", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if min(args.examples, args.frames, args.steps, args.knn) <= 0:
        raise ValueError("render and sampling counts must be positive")
    device = torch.device(args.device)
    coarse_cache = load_latent_cache(args.coarse_cache)
    fine_cache = load_latent_cache(args.fine_cache)
    if coarse_cache.names != fine_cache.names:
        raise ValueError("fine and coarse caches are not sample-aligned")
    codec = BlockPCACodec.from_state_dict(
        torch.load(args.block_codec, map_location="cpu", weights_only=False)
    )
    prior_checkpoint = torch.load(args.prior, map_location="cpu", weights_only=False)
    prior = _restore_prior(prior_checkpoint).to(device).eval()
    tokenizer_checkpoint = torch.load(
        args.tokenizer, map_location="cpu", weights_only=False
    )
    tokenizer_meta = tokenizer_checkpoint["meta"]
    spec = GridSpec(
        tuple(tokenizer_meta["grid_shape"]),
        int(tokenizer_meta["slots_per_cell"]),
    )
    tokenizer = _restore_tokenizer(tokenizer_checkpoint, spec).to(device).eval()
    feature_normalizer = FeatureNormalizer.from_state_dict(
        tokenizer_meta["normalizer"]
    )
    examples = load_fitted_corpus(args.corpus)
    held_out = set(coarse_cache.metadata["validation_sources"])
    selected = select_balanced_examples(
        [example for example in examples if example.source_id in held_out],
        args.examples,
    )
    cache_index = {name: index for index, name in enumerate(coarse_cache.names)}
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    records = []
    for example in selected:
        index = cache_index[example.name]
        condition = coarse_cache.normalized_conditions[index : index + 1].to(device)
        normalized_sample = sample_flow(
            prior,
            condition,
            batch=1,
            n_cells=coarse_cache.latents.shape[1],
            latent_dim=coarse_cache.latents.shape[2],
            device=device,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            generator=generator,
        )
        generated_codes = coarse_cache.denormalize(normalized_sample)
        generated_fine_normalized = codec.decode(generated_codes)
        generated_latent = fine_cache.denormalize(generated_fine_normalized)
        generated = feature_normalizer.denormalize(
            tokenizer.decode(generated_latent)[0]
        )
        roundtrip_codes = coarse_cache.latents[index : index + 1].to(device)
        roundtrip_fine_normalized = codec.decode(roundtrip_codes)
        roundtrip_latent = fine_cache.denormalize(roundtrip_fine_normalized)
        roundtrip = feature_normalizer.denormalize(
            tokenizer.decode(roundtrip_latent)[0]
        )
        picks = torch.linspace(0, example.shape[0] - 1, args.frames).round().long()
        target_frames = _render(
            example.features, example.shape, picks, example.background, device, args.knn
        )
        roundtrip_frames = _render(
            roundtrip, example.shape, picks, example.background, device, args.knn
        )
        generated_frames = _render(
            generated, example.shape, picks, example.background, device, args.knn
        )
        frames = [
            _row(
                [
                    _panel(target_frames[i], "held-out fitted target"),
                    _panel(roundtrip_frames[i], "16^3 hierarchy round-trip"),
                    _panel(generated_frames[i], "axial prior sample"),
                ]
            )
            for i in range(len(picks))
        ]
        path = output_dir / f"{Path(example.name).stem}_axial_comparison.gif"
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=120,
            loop=0,
        )
        record = {
            "name": example.name,
            "source_id": example.source_id,
            "target_jewels": int(example.features.shape[0]),
            "roundtrip_jewels": int(roundtrip.shape[0]),
            "generated_jewels": int(generated.shape[0]),
            "frames": picks.tolist(),
            "artifact": path.name,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    (output_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
