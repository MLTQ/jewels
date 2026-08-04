"""Render fixed block-codec held-out roundtrips beside dense fitted targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol.block_codec import BlockPCACodec
from sol.cache_latents import _restore_tokenizer
from sol.corpus import FeatureNormalizer, load_fitted_corpus
from sol.evaluation import select_balanced_examples
from sol.latent_data import load_latent_cache
from sol.render_prior_samples import _panel, _render, _row
from sol.token_grid import GridSpec


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--fine-cache", required=True)
    parser.add_argument("--coarse-cache", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--knn", type=int, default=64)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if min(args.examples, args.frames, args.knn) <= 0:
        raise ValueError("render counts must be positive")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = checkpoint["meta"]
    spec = GridSpec(tuple(meta["grid_shape"]), int(meta["slots_per_cell"]))
    model = _restore_tokenizer(checkpoint, spec).to(device).eval()
    normalizer = FeatureNormalizer.from_state_dict(meta["normalizer"])
    fine_cache = load_latent_cache(args.fine_cache)
    coarse_cache = load_latent_cache(args.coarse_cache)
    codec = BlockPCACodec.from_state_dict(
        torch.load(args.codec, map_location="cpu", weights_only=False)
    )
    reconstructed = fine_cache.denormalize(codec.decode(coarse_cache.latents))
    index_by_name = {name: index for index, name in enumerate(fine_cache.names)}
    examples = load_fitted_corpus(args.corpus)
    held_out = set(checkpoint["meta"]["validation_sources"])
    selected = select_balanced_examples(
        [example for example in examples if example.source_id in held_out],
        args.examples,
    )
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for example in selected:
        latent = reconstructed[index_by_name[example.name]][None].to(device)
        decoded = normalizer.denormalize(model.decode(latent)[0])
        picks = torch.linspace(0, example.shape[0] - 1, args.frames).round().long()
        target_frames = _render(
            example.features, example.shape, picks, example.background, device, args.knn
        )
        decoded_frames = _render(
            decoded, example.shape, picks, example.background, device, args.knn
        )
        frames = [
            _row(
                [
                    _panel(target_frames[index], "45k fitted target"),
                    _panel(decoded_frames[index], "16^3 PCA hierarchy"),
                ]
            )
            for index in range(len(picks))
        ]
        path = output_dir / f"{Path(example.name).stem}_block_roundtrip.gif"
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
            "decoded_jewels": int(decoded.shape[0]),
            "frames": picks.tolist(),
            "artifact": path.name,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    (output_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
