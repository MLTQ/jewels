"""CLI for dense latent locality, pooling, and block-PCA diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sol.latent_data import load_latent_cache
from sol.latent_hierarchy import hierarchy_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pca-blocks", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cache = load_latent_cache(args.cache)
    train_latents, _, _ = cache.split(train=True)
    grid_shape = tuple(cache.metadata["grid_shape"])
    report = hierarchy_report(
        train_latents,
        grid_shape,
        max_pca_blocks=args.max_pca_blocks,
    )
    report["cache"] = str(args.cache)
    report["tokenizer_sha256"] = cache.metadata["tokenizer_sha256"]
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
