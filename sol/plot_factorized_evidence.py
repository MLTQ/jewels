"""Plot capacity, exact-audit, progression, and jewel-size evidence for factorized v3."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re


def summary_metrics(path: Path) -> dict[str, float]:
    """Read stable held-out and structure metrics from one trainer summary."""
    summary = json.loads(path.read_text())
    evaluation = summary["latest_evaluation"]
    structure = evaluation["structure"]
    return {
        "proposals": float(summary["jewels_per_window"]),
        "psnr": float(evaluation["macro_psnr"]),
        "occupancy": float(structure["occupancy_uniformity"]),
        "active": float(structure["active_fraction"]),
        "tilt": float(structure["mixed_spacetime_tilt_median"]),
        "extent": float(structure.get("extent_median", float("nan"))),
    }


def capacity_rows(root: Path) -> list[dict[str, float]]:
    """Load the three matched proposal-capacity screens in ascending capacity."""
    rows = [
        summary_metrics(path)
        for path in root.glob("screens/v3_slots*_seed0_600/summary.json")
    ]
    if len(rows) != 3:
        raise ValueError(f"expected three capacity summaries below {root}, found {len(rows)}")
    return sorted(rows, key=lambda row: row["proposals"])


def progression_rows(path: Path, step_offset: int = 0) -> list[dict[str, float]]:
    """Load held-out evaluations, excluding incomparable minibatch training records."""
    rows = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        evaluation = record.get("evaluation")
        if evaluation is not None:
            rows.append({
                "step": float(record["step"] + step_offset),
                "psnr": float(evaluation["macro_psnr"]),
            })
    if not rows:
        raise ValueError(f"no held-out evaluation rows in {path}")
    return rows


def size_weight(path: Path) -> float:
    """Decode the registered size weight from a run-directory name."""
    if path.parent.name.startswith("control_"):
        return 0.0
    match = re.match(r"size(\d{3,4})_", path.parent.name)
    if match is None:
        raise ValueError(f"unrecognized size run {path.parent.name}")
    token = match.group(1)
    return int(token) / (10 ** (len(token) - 1))


def size_rows(root: Path) -> list[dict[str, float]]:
    """Load all matched absolute-size continuations in ascending weight."""
    paths = list((root / "size_screen").glob("*/summary.json"))
    rows = []
    for path in paths:
        row = summary_metrics(path)
        row["weight"] = size_weight(path)
        rows.append(row)
    if len(rows) < 2:
        raise ValueError(f"expected matched size summaries below {root}")
    return sorted(rows, key=lambda row: row["weight"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--teacher-extent", type=float, default=0.0128515)
    parser.add_argument("--size-offset", type=float, default=0.35)
    args = parser.parse_args()

    root = Path(args.root)
    capacities = capacity_rows(root)
    sizes = size_rows(root)
    progression = progression_rows(
        root / "selected_slots20_frozen_geometry/seed0/train_log.jsonl",
        step_offset=600,
    )
    audit = json.loads(Path(args.audit).read_text())

    import matplotlib.pyplot as plt  # noqa: PLC0415

    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))
    proposals = [row["proposals"] / 1000 for row in capacities]
    axes[0, 0].plot(proposals, [row["psnr"] for row in capacities], marker="o")
    axes[0, 0].axhline(20, color="tab:red", linestyle="--", linewidth=1)
    axes[0, 0].set(
        title="Proposal count is not the bottleneck",
        xlabel="proposals per window (thousands)", ylabel="sampled held-out PSNR (dB)",
    )

    axes[0, 1].plot(proposals, [row["active"] for row in capacities], marker="o", label="active")
    axes[0, 1].plot(proposals, [row["tilt"] for row in capacities], marker="s", label="mixed tilt")
    axes[0, 1].plot(
        proposals, [row["occupancy"] for row in capacities], marker="^", label="occupancy",
    )
    axes[0, 1].axhline(0.70, color="tab:red", linestyle="--", linewidth=1)
    axes[0, 1].axhline(0.985, color="tab:red", linestyle=":", linewidth=1)
    axes[0, 1].axhline(0.25, color="tab:red", linestyle="-.", linewidth=1)
    axes[0, 1].set(
        title="Every capacity passes structure",
        xlabel="proposals per window (thousands)", ylabel="structure statistic", ylim=(0.2, 1.01),
    )
    axes[0, 1].legend(fontsize=8)

    perceptual = audit["perceptual_macro"]
    names = ["irregular", "lattice", "teacher"]
    keys = ["irregular_seed0", "lattice", "teacher"]
    axes[1, 0].bar(names, [perceptual[key]["psnr"] for key in keys])
    axes[1, 0].axhline(20, color="tab:red", linestyle="--", linewidth=1)
    axes[1, 0].set(title="Exact audit: geometry passes, appearance fails", ylabel="PSNR (dB)")

    target = args.teacher_extent * math.exp(args.size_offset)
    for row in sizes:
        if row["weight"] == 0:
            label = "control"
        elif math.isclose(row["weight"] * 100, round(row["weight"] * 100)):
            label = f"w={row['weight']:.2f}"
        else:
            label = f"w={row['weight']:.3f}"
        passes = row["extent"] <= 0.040 and row["psnr"] >= 18.0622
        axes[1, 1].scatter(
            row["extent"], row["psnr"], s=65,
            color="tab:green" if passes else None,
        )
        axes[1, 1].annotate(label, (row["extent"], row["psnr"]), xytext=(5, 4), textcoords="offset points")
    axes[1, 1].axvline(0.040, color="tab:red", linestyle=":", linewidth=1)
    axes[1, 1].axhline(18.0622, color="tab:red", linestyle="--", linewidth=1)
    axes[1, 1].axvline(target, color="tab:green", linestyle="-.", linewidth=1, label="adjusted teacher")
    axes[1, 1].set(
        title="Absolute-size Pareto bracket",
        xlabel="median jewel extent (smaller is sharper)", ylabel="sampled held-out PSNR (dB)",
    )
    axes[1, 1].legend(fontsize=8)

    inset = axes[0, 0].inset_axes([0.53, 0.12, 0.42, 0.43])
    inset.plot([row["step"] for row in progression], [row["psnr"] for row in progression], marker="o")
    inset.set(title="appearance continuation", xlabel="total steps", ylabel="PSNR")
    inset.tick_params(labelsize=7)
    for axis in axes.flat:
        axis.grid(alpha=0.22)
    figure.tight_layout()
    figure.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
