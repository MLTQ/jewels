"""CLI for contribution-aware per-frame splat-density audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sol.corpus import load_fitted_corpus
from sol.splat_density import measure_frame_splat_density, summarize_counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", action="append", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--support-sigma", type=float, default=3.0)
    parser.add_argument("--peak-alpha", type=float, nargs="+", default=(0.01, 0.05))
    parser.add_argument("--out")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    examples = load_fitted_corpus(args.corpus, limit=args.limit)
    thresholds = tuple(sorted(set(args.peak_alpha)))
    records = []
    all_support = []
    all_effective = []
    all_thresholds: dict[float, list[torch.Tensor]] = {
        threshold: [] for threshold in thresholds
    }
    for example in examples:
        density = measure_frame_splat_density(
            example.features,
            example.shape[0],
            support_sigma=args.support_sigma,
            peak_alpha_thresholds=thresholds,
        )
        all_support.append(density.support_counts)
        all_effective.append(density.effective_peak_alpha_counts)
        for threshold, values in density.peak_alpha_counts.items():
            all_thresholds[threshold].append(values)
        records.append(
            {
                "name": example.name,
                "source_id": example.source_id,
                "domain_id": example.domain_id,
                "jewels": int(example.features.shape[0]),
                "frames": example.shape[0],
                "support_counts": summarize_counts(density.support_counts),
                "peak_alpha_counts": {
                    str(threshold): summarize_counts(values)
                    for threshold, values in density.peak_alpha_counts.items()
                },
                "effective_peak_alpha_counts": summarize_counts(
                    density.effective_peak_alpha_counts
                ),
            }
        )
    payload = {
        "corpora": args.corpus,
        "support_sigma": args.support_sigma,
        "peak_alpha_thresholds": thresholds,
        "examples": records,
        "aggregate": {
            "support_counts": summarize_counts(torch.cat(all_support)),
            "peak_alpha_counts": {
                str(threshold): summarize_counts(torch.cat(values))
                for threshold, values in all_thresholds.items()
            },
            "effective_peak_alpha_counts": summarize_counts(torch.cat(all_effective)),
        },
    }
    rendered = json.dumps(payload, indent=2)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
