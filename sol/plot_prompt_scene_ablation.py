"""Plot the causal prompt-speaker ablation from independent to shared scene state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def extract_ablation(independent: dict, shared: dict) -> dict:
    """Extract comparable controls and margins from two prompt reports."""
    rows = []
    for label, report in (("independent Jewels", independent), ("shared scene", shared)):
        controls = report["teacher_forced_controls"]
        generation = report["generation_macro"]
        rows.append({
            "label": label,
            "token_nll": {
                arm: float(controls[arm]["token_nll_macro"])
                for arm in ("correct", "shuffled", "null")
            },
            "density_nce": {
                arm: float(controls[arm]["density_nce"])
                for arm in ("correct", "shuffled", "null")
            },
            "histogram": {
                arm: float(generation[arm]["target_histogram_cosine"])
                for arm in ("correct", "shuffled", "null")
            },
            "histogram_margin_vs_null": float(
                generation["correct"]["target_histogram_cosine"]
                - generation["null"]["target_histogram_cosine"]
            ),
            "retrieval_accuracy": sum(row["correct"] for row in report["retrieval"])
            / len(report["retrieval"]),
            "gate_passed": bool(report["gate"]["passed"]),
        })
    return {
        "schema": "prompt-shared-scene-ablation-v1",
        "rows": rows,
        "shared_scene_improved_histogram_margin": (
            rows[1]["histogram_margin_vs_null"]
            > rows[0]["histogram_margin_vs_null"]
        ),
        "final_gate_passed": rows[1]["gate_passed"],
    }


def plot_ablation(report: dict, destination: Path) -> None:
    """Render controls and the causal free-generation margin in one pitch-safe figure."""
    rows = report["rows"]
    labels = [row["label"] for row in rows]
    arms = ("correct", "shuffled", "null")
    colors = {"correct": "#159947", "shuffled": "#3977c3", "null": "#e57a1f"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.24
    for index, arm in enumerate(arms):
        offset = (index - 1) * width
        axes[0, 0].bar(
            x + offset, [row["token_nll"][arm] for row in rows], width,
            label=arm, color=colors[arm],
        )
        axes[0, 1].bar(
            x + offset, [row["density_nce"][arm] for row in rows], width,
            label=arm, color=colors[arm],
        )
        axes[1, 0].bar(
            x + offset, [row["histogram"][arm] for row in rows], width,
            label=arm, color=colors[arm],
        )
    axes[0, 0].set_title("Held-out Jewel-token NLL (lower is better)")
    axes[0, 1].set_title("Held-out centroid density NCE (lower is better)")
    axes[1, 0].set_title("Free-running target histogram match (higher is better)")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, ncol=3)
    margins = [row["histogram_margin_vs_null"] for row in rows]
    bars = axes[1, 1].bar(
        x, margins, color=["#8b8b8b", "#7b4bb7"], width=0.55
    )
    axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 1].axhline(0.02, color="#cc3333", linewidth=1.5, linestyle="--", label="frozen gate")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylabel("correct − null histogram cosine")
    axes[1, 1].set_title("Shared scene reverses the free-generation margin")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, margins):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.001 if value >= 0 else -0.003),
            f"{value:+.3f}", ha="center",
        )
    figure.suptitle(
        "Native Jewel prompt speaker: causal shared-scene ablation\n"
        "source-disjoint exact prompts, two training videos per prompt",
        fontsize=16,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent", required=True)
    parser.add_argument("--shared", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    independent = json.loads(Path(args.independent).read_text())
    shared = json.loads(Path(args.shared).read_text())
    report = extract_ablation(independent, shared)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_ablation(report, output / "shared_scene_ablation.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
