"""Aggregate the preregistered shared-scene exact-prompt data scaling curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_report(value: str) -> tuple[int, Path]:
    """Parse `videos-per-prompt=report.json`."""
    count, separator, path = value.partition("=")
    if not separator or int(count) <= 0 or not path:
        raise argparse.ArgumentTypeError("reports must be COUNT=PATH")
    return int(count), Path(path)


def aggregate(points: list[tuple[int, dict]]) -> dict:
    """Extract frozen scaling signals and nondecreasing checks."""
    rows = []
    for repetitions, report in sorted(points):
        controls = report["teacher_forced_controls"]
        generation = report["generation_macro"]
        rows.append({
            "videos_per_prompt": repetitions,
            "training_fields": int(report["protocol"]["training_fields"]),
            "best_step": int(report["best_step"]),
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
            "token_margin_vs_null": float(
                controls["null"]["token_nll_macro"]
                - controls["correct"]["token_nll_macro"]
            ),
            "histogram_margin_vs_null": float(
                generation["correct"]["target_histogram_cosine"]
                - generation["null"]["target_histogram_cosine"]
            ),
            "retrieval_accuracy": sum(row["correct"] for row in report["retrieval"])
            / len(report["retrieval"]),
            "absolute_gate_passed": bool(report["gate"]["passed"]),
        })
    if len({row["videos_per_prompt"] for row in rows}) != len(rows):
        raise ValueError("shared-scene scaling points must have unique repetition counts")

    def nondecreasing(key: str) -> bool:
        values = [row[key] for row in rows]
        return all(second >= first for first, second in zip(values, values[1:]))

    checks = {
        "token_margin_vs_null_nondecreasing": nondecreasing("token_margin_vs_null"),
        "histogram_margin_vs_null_nondecreasing": nondecreasing("histogram_margin_vs_null"),
        "retrieval_accuracy_nondecreasing": nondecreasing("retrieval_accuracy"),
    }
    return {
        "schema": "shared-scene-exact-prompt-data-scaling-v1",
        "points": rows,
        "scaling_checks": checks,
        "positive_data_scaling": all(checks.values()),
        "final_absolute_gate_passed": rows[-1]["absolute_gate_passed"],
    }


def plot(report: dict, destination: Path) -> None:
    """Render prompt selectivity, free generation, and retrieval over source count."""
    points = report["points"]
    x = [row["videos_per_prompt"] for row in points]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for arm, color in (("correct", "#159947"), ("shuffled", "#3977c3"), ("null", "#e57a1f")):
        axes[0, 0].plot(x, [row["token_nll"][arm] for row in points], "o-", label=arm, color=color)
        axes[0, 1].plot(x, [row["histogram"][arm] for row in points], "o-", label=arm, color=color)
    axes[0, 0].set_title("Held-out Jewel-token NLL (lower is better)")
    axes[0, 1].set_title("Free-running target histogram match")
    axes[1, 0].plot(x, [row["token_margin_vs_null"] for row in points], "o-", label="token NLL margin")
    axes[1, 0].plot(x, [row["histogram_margin_vs_null"] for row in points], "o-", label="histogram margin")
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].axhline(0.02, color="#cc3333", linestyle="--", label="histogram gate")
    axes[1, 0].set_title("Correct advantage over prompt-blind null")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(x, [row["retrieval_accuracy"] for row in points], "o-", color="#7b4bb7")
    axes[1, 1].axhline(2 / 3, color="#cc3333", linestyle="--", label="2/3 gate")
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set_title("Prompt identity in free Jewel programs")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.set_xlabel("independent exact-prompt training videos / prompt")
        axis.set_xticks(x)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, ncol=3)
    axes[0, 1].legend(frameon=False, ncol=3)
    figure.suptitle("Shared-scene native Jewel speaker: frozen data scaling curve", fontsize=16)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=parse_report, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    points = [(count, json.loads(path.read_text())) for count, path in args.report]
    report = aggregate(points)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    plot(report, output / "shared_scene_data_scaling.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
