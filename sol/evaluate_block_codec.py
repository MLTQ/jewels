"""Held-out render evaluation for the fixed coarse block hierarchy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol.block_codec import BlockPCACodec
from sol.cache_latents import _restore_tokenizer
from sol.corpus import FeatureNormalizer, load_fitted_corpus
from sol.evaluation import EvaluationReport, ExampleMetric, select_balanced_examples
from sol.latent_data import load_latent_cache
from sol.render import render_exact
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
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if args.points <= 0 or args.max_examples <= 0:
        raise ValueError("evaluation counts must be positive")
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
    if fine_cache.names != coarse_cache.names:
        raise ValueError("fine and coarse caches are not sample-aligned")
    reconstructed_normalized = codec.decode(coarse_cache.latents)
    validation_mask = ~fine_cache.train_mask
    latent_mse = float(
        (
            reconstructed_normalized[validation_mask]
            - fine_cache.normalized_latents[validation_mask]
        )
        .square()
        .mean()
    )
    reconstructed = fine_cache.denormalize(reconstructed_normalized)
    examples = load_fitted_corpus(args.corpus)
    by_name = {example.name: example for example in examples}
    validation_examples = [
        by_name[name]
        for name, keep in zip(fine_cache.names, validation_mask, strict=True)
        if bool(keep)
    ]
    selected = select_balanced_examples(validation_examples, args.max_examples)
    index_by_name = {name: index for index, name in enumerate(fine_cache.names)}
    metrics = []
    for metric_index, example in enumerate(selected):
        latent = reconstructed[index_by_name[example.name]][None].to(device)
        decoded = normalizer.denormalize(model.decode(latent)[0])
        generator = torch.Generator(device=device).manual_seed(args.seed + metric_index)
        points = torch.rand(args.points, 3, generator=generator, device=device) * 2 - 1
        target_render = render_exact(example.features.to(device), points).clamp(0, 1)
        decoded_render = render_exact(decoded, points).clamp(0, 1)
        mse = (target_render - decoded_render).square().mean().clamp_min(1e-10)
        metrics.append(
            ExampleMetric(
                name=example.name,
                source_id=example.source_id,
                psnr=float(-10 * torch.log10(mse)),
                target_jewels=example.features.shape[0],
                decoded_jewels=decoded.shape[0],
            )
        )
    psnrs = sorted(metric.psnr for metric in metrics)
    middle = len(psnrs) // 2
    median = (
        psnrs[middle]
        if len(psnrs) % 2
        else 0.5 * (psnrs[middle - 1] + psnrs[middle])
    )
    source_values: dict[str, list[float]] = {}
    for metric in metrics:
        source_values.setdefault(metric.source_id, []).append(metric.psnr)
    source_means = {
        source: sum(values) / len(values)
        for source, values in sorted(source_values.items())
    }
    report = EvaluationReport(
        mean_psnr=sum(psnrs) / len(psnrs),
        median_psnr=median,
        macro_source_psnr=sum(source_means.values()) / len(source_means),
        source_mean_psnr=source_means,
        mean_count_ratio=sum(
            metric.decoded_jewels / metric.target_jewels for metric in metrics
        )
        / len(metrics),
        examples=tuple(metrics),
    )
    result = {
        "checkpoint": args.checkpoint,
        "fine_cache": args.fine_cache,
        "coarse_cache": args.coarse_cache,
        "codec": args.codec,
        "block_explained_variance": codec.explained_variance,
        "heldout_normalized_latent_mse": latent_mse,
        "evaluation": report.to_dict(),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
