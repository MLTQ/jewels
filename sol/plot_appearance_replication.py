"""Plot the three-seed residual-control replication screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_replication(summaries: list[Path]) -> list[dict[str, float]]:
    """Load final sampled fidelity and structure metrics in seed order."""
    rows = []
    for path in summaries:
        final = json.loads(path.read_text())["latest_evaluation"]
        structure = final["structure"]
        rows.append({
            "psnr": float(final["macro_psnr"]),
            "occupancy": float(structure["occupancy_uniformity"]),
            "active": float(structure["active_fraction"]),
            "tilt": float(structure["mixed_spacetime_tilt_median"]),
        })
    return rows


def _label(axis, bars, precision: int) -> None:
    for bar in bars:
        axis.annotate(
            f"{bar.get_height():.{precision}f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = load_replication([Path(path) for path in args.summary])
    if len(rows) != 3:
        raise ValueError("replication evidence requires exactly three seed summaries")

    import matplotlib.pyplot as plt  # noqa: PLC0415

    labels = ["seed 0", "seed 1", "seed 2"]
    colors = ["#2a6fbb", "#d97706", "#5b8c5a"]
    positions = list(range(3))
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    panels = (
        ("psnr", "Sampled PSNR", "dB", 20.0, 3),
        ("occupancy", "Occupancy uniformity", "lower is more irregular", 0.985, 5),
        ("active", "Active proposal fraction", "fraction", 0.70, 4),
        ("tilt", "Mixed spacetime tilt", "fraction", 0.25, 4),
    )
    for axis, (key, title, ylabel, threshold, precision) in zip(axes, panels):
        values = [row[key] for row in rows]
        bars = axis.bar(positions, values, color=colors)
        _label(axis, bars, precision)
        axis.axhline(threshold, color="#a61b1b", linestyle="--", linewidth=1)
        margin = max(max(values) - min(values), 0.002)
        lower = max(0.0, min(values) - 0.6 * margin)
        annotation_headroom = 0.001 if key == "occupancy" else 0.015
        upper = max(values) + max(0.6 * margin, annotation_headroom)
        lower = min(lower, threshold - 0.02 if key != "occupancy" else threshold - 0.001)
        upper = max(upper, threshold + 0.02 if key != "occupancy" else threshold + 0.001)
        axis.set(
            title=title, ylabel=ylabel,
            xticks=positions, xticklabels=labels, ylim=(lower, upper),
        )
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle(
        "Residual-control replication: total 1,200 continuation steps",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
