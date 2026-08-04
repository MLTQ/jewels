"""Render held-out target, tokenizer round-trip, and conditional prior samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import torch

from sol.autoencoder import StructuredJewelAutoencoder
from sol.corpus import FeatureNormalizer, load_fitted_corpus
from sol.evaluation import select_balanced_examples
from sol.latent_data import load_latent_cache
from sol.latent_prior import RasterFlowPrior, sample_flow
from sol.token_grid import GridSpec


def frame_points(
    shape: tuple[int, int, int], frame_indices: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    """Create production-compatible `(u,v,t)` points for selected frames only."""
    frames, height, width = shape
    times = torch.linspace(-1, 1, frames, device=device)[frame_indices.to(device)]
    vertical = torch.linspace(-1, 1, height, device=device)
    horizontal = torch.linspace(-1, 1, width, device=device)
    tt, vv, uu = torch.meshgrid(times, vertical, horizontal, indexing="ij")
    return torch.stack([uu, vv, tt], dim=-1).reshape(-1, 3)


def _production_renderer():
    stprim_root = Path(__file__).resolve().parent.parent / "stprim"
    path = str(stprim_root)
    if path not in sys.path:
        sys.path.insert(0, path)
    from models.render import render_volume
    from prior.featurize import features_to_field

    return features_to_field, render_volume


@torch.no_grad()
def _render(
    features: torch.Tensor,
    shape: tuple[int, int, int],
    picks: torch.Tensor,
    background: torch.Tensor,
    device: torch.device,
    knn: int,
) -> torch.Tensor:
    features_to_field, render_volume = _production_renderer()
    field = features_to_field(features, device=device)
    points = frame_points(shape, picks, device=device)
    output = render_volume(
        field, points, knn=knn, background=background.to(device)
    )
    return output.reshape(len(picks), shape[1], shape[2], 3).clamp(0, 1).cpu()


def _panel(frame: torch.Tensor, label: str) -> Image.Image:
    pixels = (frame * 255).round().byte().numpy()
    image = Image.fromarray(pixels)
    canvas = Image.new("RGB", (image.width, image.height + 18), "black")
    canvas.paste(image, (0, 18))
    ImageDraw.Draw(canvas).text((4, 3), label, fill="white")
    return canvas


def _row(images: list[Image.Image], pad: int = 2) -> Image.Image:
    width = sum(image.width for image in images) + pad * (len(images) - 1)
    output = Image.new("RGB", (width, max(image.height for image in images)), "white")
    offset = 0
    for image in images:
        output.paste(image, (offset, 0))
        offset += image.width + pad
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
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
    cache = load_latent_cache(args.cache)
    prior_checkpoint = torch.load(args.prior, map_location="cpu", weights_only=False)
    prior = RasterFlowPrior(**prior_checkpoint["meta"]["model_args"]).to(device)
    prior.load_state_dict(prior_checkpoint.get("ema", prior_checkpoint["model"]))
    prior.eval()
    tokenizer_checkpoint = torch.load(
        args.tokenizer, map_location="cpu", weights_only=False
    )
    tokenizer_meta = tokenizer_checkpoint["meta"]
    spec = GridSpec(
        tuple(tokenizer_meta["grid_shape"]),
        int(tokenizer_meta["slots_per_cell"]),
    )
    tokenizer = StructuredJewelAutoencoder(
        **tokenizer_meta["model_args"], spec=spec
    ).to(device)
    tokenizer.load_state_dict(tokenizer_checkpoint["model"])
    tokenizer.eval()
    feature_normalizer = FeatureNormalizer.from_state_dict(
        tokenizer_meta["normalizer"]
    )
    examples = load_fitted_corpus(args.corpus)
    held_out = set(cache.metadata["validation_sources"])
    validation = [example for example in examples if example.source_id in held_out]
    selected = select_balanced_examples(validation, args.examples)
    cache_index = {name: index for index, name in enumerate(cache.names)}
    normalized_conditions = cache.normalized_conditions
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    records = []
    for example in selected:
        index = cache_index[example.name]
        condition = normalized_conditions[index : index + 1].to(device)
        normalized_sample = sample_flow(
            prior,
            condition,
            batch=1,
            n_cells=spec.n_cells,
            latent_dim=cache.latents.shape[-1],
            device=device,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            generator=generator,
        )
        generated_latent = cache.denormalize(normalized_sample)
        generated_normalized = tokenizer.decode(generated_latent)[0]
        generated = feature_normalizer.denormalize(generated_normalized)
        target_normalized = feature_normalizer.normalize(example.features)[None].to(device)
        roundtrip_normalized = tokenizer.decode(tokenizer.encoder(target_normalized))[0]
        roundtrip = feature_normalizer.denormalize(roundtrip_normalized)
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
                    _panel(roundtrip_frames[i], "tokenizer round-trip"),
                    _panel(generated_frames[i], "conditional prior sample"),
                ]
            )
            for i in range(len(picks))
        ]
        path = output_dir / f"{Path(example.name).stem}_comparison.gif"
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=120,
            loop=0,
        )
        records.append(
            {
                "name": example.name,
                "source_id": example.source_id,
                "target_jewels": int(example.features.shape[0]),
                "roundtrip_jewels": int(roundtrip.shape[0]),
                "generated_jewels": int(generated.shape[0]),
                "frames": picks.tolist(),
                "artifact": path.name,
            }
        )
        print(json.dumps(records[-1]), flush=True)
    (output_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
