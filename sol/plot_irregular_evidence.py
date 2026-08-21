"""Plot the preregistered irregular-field training and matched-control evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def evaluation_rows(path: Path, step_offset: int = 0) -> list[dict[str, float]]:
    """Load only held-out evaluation records from a JSONL training log."""
    rows = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        evaluation = record.get("evaluation")
        if evaluation is None:
            continue
        structure = evaluation["structure"]
        rows.append({
            "step": float(record["step"] + step_offset),
            "psnr": float(evaluation["macro_psnr"]),
            "occupancy": float(structure["occupancy_uniformity"]),
            "active": float(structure["active_fraction"]),
            "mixed_tilt": float(
                structure.get("mixed_spacetime_tilt_median", float("nan"))
            ),
        })
    if not rows:
        raise ValueError(f"no evaluation records in {path}")
    return rows


def summary_point(path: Path) -> tuple[float, float]:
    """Return held-out PSNR and occupancy from one run summary."""
    evaluation = json.loads(path.read_text())["latest_evaluation"]
    return (
        float(evaluation["macro_psnr"]),
        float(evaluation["structure"]["occupancy_uniformity"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    runs = {
        "appearance-biased": evaluation_rows(
            root / "weak_density" / "train_log.jsonl"
        ),
        "direct spacetime tilt": evaluation_rows(
            root / "direct_tilt" / "train_log.jsonl"
        ),
        "geometry frozen after 2k": evaluation_rows(
            root / "frozen_geometry" / "train_log.jsonl", step_offset=2000
        ),
    }
    audit = json.loads(Path(args.audit).read_text())
    matched = {
        "lattice": (
            audit["perceptual_macro"]["lattice"]["psnr"],
            audit["structure_macro"]["lattice"]["occupancy_uniformity"],
        ),
        "dense 200": summary_point(root / "dense_200" / "summary.json"),
        "sparse 200": summary_point(root / "sparse_200" / "summary.json"),
        "tilt 6000": summary_point(root / "direct_tilt" / "summary.json"),
        "frozen 6000": summary_point(root / "frozen_geometry" / "summary.json"),
    }

    import matplotlib.pyplot as plt  # noqa: PLC0415

    figure, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    colors = {
        "appearance-biased": "tab:blue",
        "direct spacetime tilt": "tab:orange",
        "geometry frozen after 2k": "tab:green",
    }
    for label, rows in runs.items():
        steps = [row["step"] for row in rows]
        axes[0].plot(
            steps, [row["psnr"] for row in rows], marker="o",
            label=label, color=colors[label],
        )
        axes[1].plot(
            steps, [row["occupancy"] for row in rows], marker="o",
            label=label, color=colors[label],
        )
        axes[2].plot(
            steps, [row["active"] for row in rows], marker="o",
            label=f"{label}: active", color=colors[label],
        )
        tilts = [row["mixed_tilt"] for row in rows]
        if not all(math.isnan(value) for value in tilts):
            axes[2].plot(
                steps, tilts, marker="s", linestyle="--",
                label=f"{label}: mixed axis", color=colors[label],
            )

    axes[0].axhline(20, color="tab:red", linestyle="--", linewidth=1)
    axes[0].set(title="Held-out fidelity", xlabel="training step", ylabel="PSNR (dB)")
    axes[1].axhline(0.985, color="tab:red", linestyle="--", linewidth=1)
    axes[1].set(
        title="Grid suppression", xlabel="training step",
        ylabel="occupancy uniformity", ylim=(0.975, 0.993),
    )
    axes[2].axhline(0.70, color="tab:red", linestyle="--", linewidth=1)
    axes[2].axhline(0.25, color="tab:red", linestyle=":", linewidth=1)
    axes[2].set(
        title="Sparsity and time distortion", xlabel="training step",
        ylabel="fraction / mixed tilt", ylim=(0, 0.75),
    )
    axes[2].legend(fontsize=7)

    for label, (psnr, occupancy) in matched.items():
        axes[3].scatter(occupancy, psnr, s=55)
        axes[3].annotate(label, (occupancy, psnr), xytext=(4, 4), textcoords="offset points")
    axes[3].axvline(0.985, color="tab:red", linestyle="--", linewidth=1)
    axes[3].axhline(20, color="tab:red", linestyle="--", linewidth=1)
    axes[3].set(
        title="Quality–irregularity tradeoff", xlabel="occupancy uniformity",
        ylabel="held-out PSNR (dB)", xlim=(0.972, 1.001), ylim=(15, 25),
    )

    axes[0].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.tight_layout()
    figure.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
