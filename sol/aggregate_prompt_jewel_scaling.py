"""Aggregate preregistered additive prompt-to-Jewel data-scaling evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _strictly_ordered(values: list[int]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def _nondecreasing(values: list[float], *, tolerance: float = 1e-9) -> bool:
    return all(
        right + tolerance >= left for left, right in zip(values, values[1:])
    )


def summarize_point(path: Path, label: str) -> dict:
    """Reduce one additive-caster report to the preregistered curve metrics."""
    report = json.loads(path.read_text())
    if report.get("schema") != "additive-prompt-native-jewel-caster-gate-v1":
        raise ValueError(f"{path} is not an additive prompt-caster report")
    controls = report["teacher_forced_controls"]
    generation = report["generation_macro"]
    correct = controls["correct"]
    retrieval = report["retrieval"]
    return {
        "label": label,
        "report": str(path),
        "training_fields": int(report["protocol"]["training_fields"]),
        "cell_nll": {arm: float(controls[arm]["cell_nll"]) for arm in controls},
        "token_nll_macro": {
            arm: float(controls[arm]["token_nll_macro"]) for arm in controls
        },
        "cell_margin_vs_shuffled": float(
            controls["shuffled"]["cell_nll"] - correct["cell_nll"]
        ),
        "cell_margin_vs_null": float(
            controls["null"]["cell_nll"] - correct["cell_nll"]
        ),
        "token_margin_vs_shuffled": float(
            controls["shuffled"]["token_nll_macro"]
            - correct["token_nll_macro"]
        ),
        "token_margin_vs_null": float(
            controls["null"]["token_nll_macro"] - correct["token_nll_macro"]
        ),
        "target_histogram_cosine": {
            arm: float(generation[arm]["target_histogram_cosine"])
            for arm in generation
        },
        "retrieval_accuracy": sum(bool(row["correct"]) for row in retrieval)
        / len(retrieval),
        "absolute_gate_passed": bool(report["gate"]["passed"]),
    }


def aggregate(points: list[dict]) -> dict:
    """Sort curve points and apply the frozen monotonic-scaling checks."""
    if len(points) < 2:
        raise ValueError("a scaling curve needs at least two reports")
    points = sorted(points, key=lambda row: row["training_fields"])
    counts = [row["training_fields"] for row in points]
    if not _strictly_ordered(counts):
        raise ValueError("training-field counts must be unique")
    checks = {
        "cell_margin_vs_shuffled_nondecreasing": _nondecreasing(
            [row["cell_margin_vs_shuffled"] for row in points]
        ),
        "token_margin_vs_shuffled_nondecreasing": _nondecreasing(
            [row["token_margin_vs_shuffled"] for row in points]
        ),
        "correct_histogram_cosine_nondecreasing": _nondecreasing(
            [row["target_histogram_cosine"]["correct"] for row in points]
        ),
        "retrieval_accuracy_nondecreasing": _nondecreasing(
            [row["retrieval_accuracy"] for row in points]
        ),
    }
    return {
        "schema": "prompt-to-jewel-data-scaling-v1",
        "points": points,
        "scaling_checks": checks,
        "positive_data_scaling": all(checks.values()),
        "final_absolute_gate_passed": points[-1]["absolute_gate_passed"],
    }


def plot(report: dict, output: Path) -> None:
    """Render the four-panel prompt-scaling evidence figure."""
    import matplotlib.pyplot as plt

    points = report["points"]
    x = [row["training_fields"] for row in points]
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for key, label, color in (
        ("cell_margin_vs_shuffled", "vs shuffled prompt", "tab:blue"),
        ("cell_margin_vs_null", "vs prompt-blind prior", "tab:orange"),
    ):
        axes[0, 0].plot(x, [row[key] for row in points], "o-", label=label, color=color)
    axes[0, 0].axhline(0, color="black", linewidth=1)
    axes[0, 0].set(title="Centroid language advantage", xlabel="training videos", ylabel="control NLL - correct NLL")
    axes[0, 0].legend()

    for key, label, color in (
        ("token_margin_vs_shuffled", "vs shuffled prompt", "tab:blue"),
        ("token_margin_vs_null", "vs prompt-blind prior", "tab:orange"),
    ):
        axes[0, 1].plot(x, [row[key] for row in points], "o-", label=label, color=color)
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].set(title="Jewel-token language advantage", xlabel="training videos", ylabel="control NLL - correct NLL")
    axes[0, 1].legend()

    for arm, label, color in (
        ("correct", "correct prompt", "tab:green"),
        ("shuffled", "shuffled prompt", "tab:blue"),
        ("null", "prompt-blind prior", "tab:orange"),
    ):
        axes[1, 0].plot(
            x,
            [row["target_histogram_cosine"][arm] for row in points],
            "o-", label=label, color=color,
        )
    axes[1, 0].set(title="Free-running field match", xlabel="training videos", ylabel="target histogram cosine")
    axes[1, 0].legend()

    accuracy = [row["retrieval_accuracy"] for row in points]
    axes[1, 1].plot(x, accuracy, "o-", color="tab:purple", label="top-1 retrieval")
    axes[1, 1].axhline(2 / 3, color="tab:red", linestyle="--", linewidth=1, label="2/3 gate")
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set(title="Prompt identity in free samples", xlabel="training videos", ylabel="retrieval accuracy")
    axes[1, 1].legend()
    figure.suptitle("Prompt-to-Jewel data scaling (frozen language and speaker)", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", action="append", required=True,
        help="label=path/to/additive/report.json",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    points = []
    for value in args.report:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("each --report must be label=path")
        points.append(summarize_point(Path(raw_path), label))
    report = aggregate(points)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    plot(report, output / "prompt_data_scaling.png")
    print(json.dumps({
        "positive_data_scaling": report["positive_data_scaling"],
        "final_absolute_gate_passed": report["final_absolute_gate_passed"],
        "points": report["points"],
    }, indent=2))


if __name__ == "__main__":
    main()
