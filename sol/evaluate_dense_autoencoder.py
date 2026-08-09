"""Evaluate a sparse dense-jewel tokenizer checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol.corpus import FeatureNormalizer, load_fitted_corpus
from sol.evaluation import evaluate_roundtrip
from sol.token_grid import GridSpec
from sol.tokenizer_checkpoint import build_tokenizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all-sources", action="store_true")
    parser.add_argument("--slots-override", type=int, default=0)
    parser.add_argument("--out")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.points <= 0 or args.max_examples <= 0:
        raise ValueError("evaluation counts must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = checkpoint["meta"]
    trained_slots = int(meta["slots_per_cell"])
    if args.slots_override and args.slots_override < trained_slots:
        raise ValueError("slots override cannot reduce checkpoint capacity")
    spec = GridSpec(
        tuple(meta["grid_shape"]), args.slots_override or trained_slots
    )
    model = build_tokenizer(meta, spec).to(args.device)
    model.load_state_dict(checkpoint["model"])
    normalizer = FeatureNormalizer.from_state_dict(meta["normalizer"])
    corpus_path = args.corpus or meta["corpus"]
    examples = load_fitted_corpus(corpus_path)
    held_out = set(meta["validation_sources"])
    validation = (
        examples
        if args.all_sources
        else [example for example in examples if example.source_id in held_out]
    )
    if not validation:
        raise ValueError(
            "no tokenizer-held-out sources occur in this corpus; use --all-sources "
            "for an explicitly cross-domain transfer audit"
        )
    report = evaluate_roundtrip(
        model,
        validation,
        normalizer,
        device=args.device,
        points_per_example=args.points,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "step": int(checkpoint["step"]),
        "corpus": str(corpus_path),
        "validation_sources": sorted(held_out),
        "selection": "all_corpus_sources" if args.all_sources else "tokenizer_validation_sources",
        "trained_slots_per_cell": trained_slots,
        "evaluated_slots_per_cell": spec.slots_per_cell,
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
