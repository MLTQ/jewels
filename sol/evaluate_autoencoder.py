"""Evaluate a trained structured jewel autoencoder checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol.autoencoder import StructuredJewelAutoencoder
from sol.corpus import FeatureNormalizer, load_fitted_corpus
from sol.evaluation import evaluate_roundtrip
from sol.token_grid import GridSpec


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.points <= 0 or args.max_examples <= 0:
        raise ValueError("evaluation counts must be positive")
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    meta = checkpoint["meta"]
    spec = GridSpec(
        tuple(meta["grid_shape"]), int(meta["slots_per_cell"])
    )
    model = StructuredJewelAutoencoder(
        **meta["model_args"], spec=spec
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    normalizer = FeatureNormalizer.from_state_dict(meta["normalizer"])
    corpus_path = args.corpus or meta["corpus"]
    examples = load_fitted_corpus(corpus_path)
    held_out = set(meta["validation_sources"])
    validation = [example for example in examples if example.source_id in held_out]
    observed_sources = {example.source_id for example in validation}
    if observed_sources != held_out:
        raise ValueError(
            f"checkpoint expects held-out sources {sorted(held_out)}, "
            f"found {sorted(observed_sources)}"
        )
    report = evaluate_roundtrip(
        model,
        validation,
        normalizer,
        device=device,
        points_per_example=args.points,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "step": int(checkpoint["step"]),
        "corpus": str(corpus_path),
        "validation_sources": sorted(held_out),
        "points_per_example": args.points,
        "requested_max_examples": args.max_examples,
        "evaluation": report.to_dict(),
    }
    rendered = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
