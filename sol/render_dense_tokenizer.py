"""Render held-out dense fitted targets beside sparse tokenizer round-trips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol.corpus import FeatureNormalizer, load_fitted_corpus
from sol.evaluation import select_balanced_examples
from sol.render_prior_samples import _panel, _render, _row
from sol.sparse_autoencoder import SparseJewelAutoencoder
from sol.token_grid import GridSpec


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--knn", type=int, default=64)
    parser.add_argument("--names", nargs="*", default=())
    parser.add_argument("--slots-override", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if min(args.examples, args.frames, args.knn) <= 0:
        raise ValueError("render counts must be positive")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = checkpoint["meta"]
    trained_slots = int(meta["slots_per_cell"])
    if args.slots_override and args.slots_override < trained_slots:
        raise ValueError("slots override cannot reduce checkpoint capacity")
    spec = GridSpec(
        tuple(meta["grid_shape"]), args.slots_override or trained_slots
    )
    model = SparseJewelAutoencoder(**meta["model_args"], spec=spec).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalizer = FeatureNormalizer.from_state_dict(meta["normalizer"])
    examples = load_fitted_corpus(args.corpus)
    held_out = set(meta["validation_sources"])
    if args.names:
        requested = set(args.names)
        selected = [example for example in examples if example.name in requested]
        missing = requested - {example.name for example in selected}
        if missing:
            raise ValueError(f"unknown requested examples: {sorted(missing)}")
    else:
        validation = [example for example in examples if example.source_id in held_out]
        selected = select_balanced_examples(validation, args.examples)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for example in selected:
        normalized = normalizer.normalize(example.features)[None].to(device)
        decoded = normalizer.denormalize(model.decode(model.encoder(normalized))[0])
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
                    _panel(
                        target_frames[index],
                        f"{example.features.shape[0]:,}-jewel fitted target",
                    ),
                    _panel(decoded_frames[index], "sparse tokenizer round-trip"),
                ]
            )
            for index in range(len(picks))
        ]
        path = output_dir / f"{Path(example.name).stem}_dense_roundtrip.gif"
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
            "decoded_jewels": int(decoded.shape[0]),
            "trained_slots_per_cell": trained_slots,
            "rendered_slots_per_cell": spec.slots_per_cell,
            "frames": picks.tolist(),
            "artifact": path.name,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    (output_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
