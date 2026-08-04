"""Evaluate a saved raster-flow prior on the frozen held-out latent protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol.axial_prior import AxialFlowPrior
from sol.latent_data import load_latent_cache
from sol.latent_prior import RasterFlowPrior
from sol.prior_evaluation import evaluate_prior


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    return parser.parse_args()


def _restore_prior(checkpoint: dict) -> torch.nn.Module:
    architecture = checkpoint["meta"].get("architecture", "raster_flow_v1")
    model_class = AxialFlowPrior if architecture.startswith("axial_flow") else RasterFlowPrior
    model = model_class(**checkpoint["meta"]["model_args"])
    model.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
    return model


def main() -> None:
    args = _parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cache_path = args.cache or checkpoint["meta"]["cache"]
    cache = load_latent_cache(cache_path)
    model = _restore_prior(checkpoint).to(args.device)
    train_latents, train_conditions, _ = cache.split(train=True)
    validation_latents, validation_conditions, validation_indices = cache.split(
        train=False
    )
    report = evaluate_prior(
        model,
        train_latents,
        train_conditions,
        validation_latents,
        validation_conditions,
        device=args.device,
        sample_steps=args.sample_steps,
        sample_examples=args.samples,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "step": int(checkpoint["step"]),
        "cache": str(cache_path),
        "architecture": checkpoint["meta"].get("architecture", "raster_flow_v1"),
        "sample_steps": args.sample_steps,
        "cfg_scale": args.cfg_scale,
        "validation_names": [cache.names[int(index)] for index in validation_indices],
        "evaluation": report.to_dict(),
    }
    rendered = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
