"""Render matched held-out comparisons for two tokenizer checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol.autoencoder import StructuredJewelAutoencoder
from sol.corpus import FeatureNormalizer, FittedExample, load_fitted_corpus
from sol.evaluation import select_balanced_examples
from sol.render_prior_samples import _panel, _render, _row
from sol.token_grid import GridSpec


def _load_tokenizer(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    meta = checkpoint["meta"]
    spec = GridSpec(tuple(meta["grid_shape"]), int(meta["slots_per_cell"]))
    model = StructuredJewelAutoencoder(**meta["model_args"], spec=spec).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, FeatureNormalizer.from_state_dict(meta["normalizer"]), meta


@torch.no_grad()
def _roundtrip(
    model: StructuredJewelAutoencoder,
    normalizer: FeatureNormalizer,
    example: FittedExample,
    device: torch.device,
) -> torch.Tensor:
    normalized = normalizer.normalize(example.features)[None].to(device)
    decoded = model.decode(model.encoder(normalized))[0]
    return normalizer.denormalize(decoded)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--corpus", required=True)
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
        raise ValueError("comparison counts must be positive")
    device = torch.device(args.device)
    baseline, baseline_normalizer, baseline_meta = _load_tokenizer(
        args.baseline, device
    )
    candidate, candidate_normalizer, candidate_meta = _load_tokenizer(
        args.candidate, device
    )
    baseline_sources = set(baseline_meta["validation_sources"])
    candidate_sources = set(candidate_meta["validation_sources"])
    if baseline_sources != candidate_sources:
        raise ValueError("tokenizers must use the same held-out sources")
    examples = load_fitted_corpus(args.corpus)
    validation = [example for example in examples if example.source_id in baseline_sources]
    selected = select_balanced_examples(validation, args.examples)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for example in selected:
        baseline_features = _roundtrip(
            baseline, baseline_normalizer, example, device
        )
        candidate_features = _roundtrip(
            candidate, candidate_normalizer, example, device
        )
        picks = torch.linspace(0, example.shape[0] - 1, args.frames).round().long()
        target_frames = _render(
            example.features, example.shape, picks, example.background, device, args.knn
        )
        baseline_frames = _render(
            baseline_features,
            example.shape,
            picks,
            example.background,
            device,
            args.knn,
        )
        candidate_frames = _render(
            candidate_features,
            example.shape,
            picks,
            example.background,
            device,
            args.knn,
        )
        frames = [
            _row(
                [
                    _panel(target_frames[index], "held-out fitted target"),
                    _panel(baseline_frames[index], "5.15x tokenizer"),
                    _panel(candidate_frames[index], "2.57x tokenizer"),
                ]
            )
            for index in range(len(picks))
        ]
        path = output_dir / f"{Path(example.name).stem}_tokenizers.gif"
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=120,
            loop=0,
        )
        record = {
            "name": example.name,
            "target_jewels": int(example.features.shape[0]),
            "baseline_jewels": int(baseline_features.shape[0]),
            "candidate_jewels": int(candidate_features.shape[0]),
            "frames": picks.tolist(),
            "artifact": path.name,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    (output_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
