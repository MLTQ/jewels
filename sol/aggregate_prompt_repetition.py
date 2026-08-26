"""Aggregate the exact-prompt source-repetition Gate 1e curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(path: Path, repetition: int) -> dict:
    """Extract controlled prompt metrics from one factorized neural report."""
    report = json.loads(path.read_text())
    if report.get("schema") != "factorized-prompt-native-jewel-caster-gate-v1":
        raise ValueError("exact-prompt repetition needs a factorized prompt report")
    controls = report["teacher_forced_controls"]
    generation = report["generation_macro"]
    correct = controls["correct"]
    return {
        "repetitions_per_prompt": repetition,
        "report": str(path),
        "training_fields": int(report["protocol"]["training_fields"]),
        "best_step": int(report["best_step"]),
        "density_nce": {
            arm: float(controls[arm]["density_nce"]) for arm in controls
        },
        "token_nll_macro": {
            arm: float(controls[arm]["token_nll_macro"]) for arm in controls
        },
        "density_margin_vs_shuffled": float(
            controls["shuffled"]["density_nce"] - correct["density_nce"]
        ),
        "density_margin_vs_null": float(
            controls["null"]["density_nce"] - correct["density_nce"]
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
        "retrieval_accuracy": sum(bool(row["correct"]) for row in report["retrieval"])
        / len(report["retrieval"]),
        "absolute_gate_passed": bool(report["gate"]["passed"]),
    }


def _nondecreasing(values: list[float], *, tolerance: float = 1e-9) -> bool:
    return all(
        right + tolerance >= left for left, right in zip(values, values[1:])
    )


def aggregate(points: list[dict]) -> dict:
    """Apply the frozen repetition-scaling checks and final absolute verdict."""
    if len(points) < 2:
        raise ValueError("repetition scaling needs at least two points")
    points = sorted(points, key=lambda row: row["repetitions_per_prompt"])
    repetitions = [row["repetitions_per_prompt"] for row in points]
    if any(left >= right for left, right in zip(repetitions, repetitions[1:])):
        raise ValueError("repetition counts must be unique and increasing")
    checks = {
        "density_margin_vs_null_nondecreasing": _nondecreasing(
            [row["density_margin_vs_null"] for row in points]
        ),
        "token_margin_vs_null_nondecreasing": _nondecreasing(
            [row["token_margin_vs_null"] for row in points]
        ),
        "correct_histogram_cosine_nondecreasing": _nondecreasing(
            [row["target_histogram_cosine"]["correct"] for row in points]
        ),
        "retrieval_accuracy_nondecreasing": _nondecreasing(
            [row["retrieval_accuracy"] for row in points]
        ),
    }
    return {
        "schema": "exact-prompt-source-repetition-v1",
        "points": points,
        "scaling_checks": checks,
        "positive_repetition_scaling": all(checks.values()),
        "final_absolute_gate_passed": points[-1]["absolute_gate_passed"],
    }


def plot(report: dict, output: Path) -> None:
    """Render likelihood, free-run, and retrieval changes with exact-prompt repetition."""
    import matplotlib.pyplot as plt

    points = report["points"]
    x = [row["repetitions_per_prompt"] for row in points]
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for key, label, color in (
        ("density_margin_vs_shuffled", "vs shuffled", "tab:blue"),
        ("density_margin_vs_null", "vs null", "tab:orange"),
    ):
        axes[0, 0].plot(x, [row[key] for row in points], "o-", label=label, color=color)
    axes[0, 0].axhline(0, color="black", linewidth=1)
    axes[0, 0].set(title="Centroid-density advantage", xlabel="exact-prompt training videos / prompt", ylabel="control NCE - correct NCE")
    axes[0, 0].legend()
    for key, label, color in (
        ("token_margin_vs_shuffled", "vs shuffled", "tab:blue"),
        ("token_margin_vs_null", "vs null", "tab:orange"),
    ):
        axes[0, 1].plot(x, [row[key] for row in points], "o-", label=label, color=color)
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].set(title="Jewel-token advantage", xlabel="exact-prompt training videos / prompt", ylabel="control NLL - correct NLL")
    axes[0, 1].legend()
    for arm, label, color in (
        ("correct", "correct", "tab:green"),
        ("shuffled", "shuffled", "tab:blue"),
        ("null", "null", "tab:orange"),
    ):
        axes[1, 0].plot(x, [row["target_histogram_cosine"][arm] for row in points], "o-", label=label, color=color)
    axes[1, 0].set(title="Free-running field match", xlabel="exact-prompt training videos / prompt", ylabel="target histogram cosine")
    axes[1, 0].legend()
    axes[1, 1].plot(x, [row["retrieval_accuracy"] for row in points], "o-", color="tab:purple", label="top-1 retrieval")
    axes[1, 1].axhline(2 / 3, color="tab:red", linestyle="--", label="2/3 gate")
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set(title="Prompt identity in free samples", xlabel="exact-prompt training videos / prompt", ylabel="retrieval accuracy")
    axes[1, 1].legend()
    figure.suptitle("Exact-prompt source repetition (frozen Jewel language and speaker)", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, help="repetitions=report.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    points = []
    for value in args.report:
        raw_repetition, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError("each report must be repetitions=path")
        points.append(summarize(Path(raw_path), int(raw_repetition)))
    report = aggregate(points)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    plot(report, output / "exact_prompt_repetition.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
