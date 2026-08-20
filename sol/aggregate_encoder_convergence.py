"""Aggregate encoder convergence runs and draw the primary scaling evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def confidence(values: list[float]) -> dict[str, float]:
    """Return mean, sample spread, and a 95% t interval for three seeds."""
    mean = statistics.mean(values)
    if len(values) < 2:
        return {"mean": mean, "sd": 0.0, "ci95_low": mean, "ci95_high": mean}
    sd = statistics.stdev(values)
    critical = 4.303 if len(values) == 3 else 1.96
    radius = critical * sd / math.sqrt(len(values))
    return {"mean": mean, "sd": sd, "ci95_low": mean - radius,
            "ci95_high": mean + radius}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.out)
    protocol = json.loads((root / "protocol.json").read_text())
    runs = []
    curves: dict[int, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for size in protocol["train_sizes"]:
        for seed in protocol["seeds"]:
            run = root / f"n{size}" / f"seed{seed}"
            summary = json.loads((run / "summary.json").read_text())
            evaluation = summary["best_evaluation"]
            styles: dict[str, list[float]] = defaultdict(list)
            for source_id, row in evaluation.items():
                if source_id == "macro_psnr":
                    continue
                styles[source_id.split("__", 1)[0]].append(row["psnr"])
            runs.append({
                "train_size": size, "seed": seed,
                "best_epoch": summary["best_epoch"],
                "trained_epochs": summary["epochs"],
                "stopped_early": summary["stopped_early"],
                "macro_psnr": evaluation["macro_psnr"],
                "per_style_psnr": {key: statistics.mean(value)
                                    for key, value in sorted(styles.items())},
            })
            for line in (run / "train_log.jsonl").read_text().splitlines():
                record = json.loads(line)
                if "evaluation" in record:
                    curves[size][float(record["epoch"])].append(
                        record["evaluation"]["macro_psnr"]
                    )
    sizes = protocol["train_sizes"]
    scaling = {
        str(size): confidence([run["macro_psnr"] for run in runs
                               if run["train_size"] == size])
        for size in sizes
    }
    report = {
        "schema": "encoder-convergence-report-v2",
        "protocol": protocol,
        "scaling": scaling,
        "largest_two_delta_db": scaling[str(sizes[-1])]["mean"]
        - scaling[str(sizes[-2])]["mean"],
        "runs": runs,
        "curves": {
            str(size): {str(epoch): confidence(values)
                        for epoch, values in sorted(points.items())}
            for size, points in curves.items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib.pyplot as plt  # noqa: PLC0415

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    means = [scaling[str(size)]["mean"] for size in sizes]
    low = [means[i] - scaling[str(size)]["ci95_low"] for i, size in enumerate(sizes)]
    high = [scaling[str(size)]["ci95_high"] - means[i] for i, size in enumerate(sizes)]
    axes[0].errorbar(sizes, means, yerr=[low, high], marker="o", capsize=5)
    axes[0].set_xscale("log")
    axes[0].set_xticks(sizes, labels=[str(size) for size in sizes])
    axes[0].set_xlabel("Training videos")
    axes[0].set_ylabel("Best held-out PSNR (dB)")
    axes[0].set_title("Data scaling (mean ± 95% CI, 3 seeds)")
    for size, value in zip(sizes, means):
        axes[0].annotate(f"{value:.2f}", (size, value), xytext=(0, 8),
                         textcoords="offset points", ha="center")
    for size in sizes:
        epochs = sorted(curves[size])
        axes[1].plot(epochs, [statistics.mean(curves[size][epoch]) for epoch in epochs],
                     marker="o", label=f"n={size}")
    axes[1].set_xlabel("Full corpus passes (epochs)")
    axes[1].set_ylabel("Held-out PSNR (dB)")
    axes[1].set_title("Matched-exposure convergence")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "convergence.png", dpi=180)
    print(json.dumps({"scaling": scaling,
                      "largest_two_delta_db": report["largest_two_delta_db"]}, indent=2))


if __name__ == "__main__":
    main()
