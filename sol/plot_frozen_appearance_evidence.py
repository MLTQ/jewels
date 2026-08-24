"""Plot frozen-geometry fidelity scaling, replication, and stability ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _last_record(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    training = [record for record in records if "render_loss" in record]
    if not training:
        raise ValueError(f"training log contains no metric records: {path}")
    return training[-1]


def _temporal_deviation(report: dict, arm: str) -> float:
    rows = [row for row in report["perceptual_records"] if row["arm"] == arm]
    if not rows:
        raise ValueError(f"audit contains no records for {arm}")
    return statistics.mean(
        abs(row["render_signature"]["temporal_change_ratio"] - 1.0)
        for row in rows
    )


def collect_evidence(root: Path) -> dict[str, object]:
    """Collect registered metrics under stable scientific labels."""
    ablation = _read(root / "audit_seed0" / "report.json")
    replication = _read(root / "replication" / "audit_final_seeds" / "report.json")
    ablation_labels = {
        "source": "irregular_seed0",
        "render": "irregular_seed1",
        "perceptual": "irregular_seed2",
        "stabilized": "irregular_seed3",
    }
    replication_labels = (
        "irregular_seed1", "irregular_seed2", "irregular_seed3"
    )
    macro = ablation["perceptual_macro"]
    final_macro = replication["perceptual_macro"]
    compute = {
        "labels": ["frozen source", "+600 render", "+1,200 render"],
        "psnr": [
            macro[ablation_labels["source"]]["psnr"],
            macro[ablation_labels["render"]]["psnr"],
            final_macro[replication_labels[0]]["psnr"],
        ],
        "lpips": [
            macro[ablation_labels["source"]]["lpips"],
            macro[ablation_labels["render"]]["lpips"],
            final_macro[replication_labels[0]]["lpips"],
        ],
    }
    replication_metrics = {
        "psnr": [final_macro[label]["psnr"] for label in replication_labels],
        "lpips": [final_macro[label]["lpips"] for label in replication_labels],
    }
    render_macro = macro[ablation_labels["render"]]
    ablation_delta = {
        "labels": ["perceptual", "stabilized"],
        "psnr": [
            macro[ablation_labels[name]]["psnr"] - render_macro["psnr"]
            for name in ("perceptual", "stabilized")
        ],
        "lpips_improvement": [
            render_macro["lpips"] - macro[ablation_labels[name]]["lpips"]
            for name in ("perceptual", "stabilized")
        ],
    }
    stability = {"labels": ["render", "perceptual", "stabilized"]}
    stability["temporal_deviation"] = [
        _temporal_deviation(ablation, ablation_labels[name])
        for name in stability["labels"]
    ]
    logs = root / "screens"
    stability["out_of_range"] = [
        _last_record(logs / directory / "train_log.jsonl")[
            "render_out_of_range_fraction"
        ]
        for directory in (
            "frozen_render_seed0_600",
            "frozen_perceptual_seed0_600",
            "frozen_stabilized_seed0_600",
        )
    ]
    return {
        "compute": compute,
        "replication": replication_metrics,
        "ablation_delta": ablation_delta,
        "stability": stability,
    }


def plot_evidence(evidence: dict[str, object], output: Path) -> None:
    """Render one pitch-readable four-panel frozen-appearance figure."""
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    blue, orange, green = "#2878B5", "#F28E2B", "#59A14F"

    compute = evidence["compute"]
    x = np.arange(3)
    axes[0, 0].plot(x, compute["psnr"], marker="o", color=blue, linewidth=2)
    axes[0, 0].axhline(20, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_xticks(x, compute["labels"], rotation=12)
    axes[0, 0].set_ylabel("exact PSNR (dB)", color=blue)
    lpips_axis = axes[0, 0].twinx()
    lpips_axis.plot(x, compute["lpips"], marker="s", color=orange, linewidth=2)
    lpips_axis.set_ylabel("LPIPS (lower is better)", color=orange)
    axes[0, 0].set_title("Frozen geometry: compute scaling")

    replication = evidence["replication"]
    seeds = np.arange(3)
    axes[0, 1].bar(seeds - 0.16, replication["psnr"], 0.32, color=blue)
    axes[0, 1].axhline(20, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_ylim(19.95, max(replication["psnr"]) + 0.03)
    axes[0, 1].set_xticks(seeds, ("seed 0", "seed 1", "seed 2"))
    axes[0, 1].set_ylabel("exact PSNR (dB)", color=blue)
    replication_lpips = axes[0, 1].twinx()
    replication_lpips.plot(
        seeds, replication["lpips"], marker="s", color=orange, linewidth=2
    )
    replication_lpips.set_ylabel("LPIPS", color=orange)
    axes[0, 1].set_title("20 dB crossing replicates")

    delta = evidence["ablation_delta"]
    choices = np.arange(2)
    axes[1, 0].bar(choices - 0.18, delta["psnr"], 0.36, color=blue)
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set_xticks(choices, delta["labels"])
    axes[1, 0].set_ylabel("PSNR delta vs render (dB)", color=blue)
    delta_lpips = axes[1, 0].twinx()
    delta_lpips.bar(
        choices + 0.18, delta["lpips_improvement"], 0.36, color=orange
    )
    delta_lpips.set_ylabel("LPIPS improvement vs render", color=orange)
    axes[1, 0].set_title("Tested perceptual objective is dominated")

    stability = evidence["stability"]
    arms = np.arange(3)
    axes[1, 1].bar(
        arms - 0.18, stability["temporal_deviation"], 0.36, color=green
    )
    axes[1, 1].set_xticks(arms, stability["labels"], rotation=10)
    axes[1, 1].set_ylabel("mean |temporal ratio - 1|", color=green)
    range_axis = axes[1, 1].twinx()
    range_axis.bar(
        arms + 0.18,
        np.asarray(stability["out_of_range"]) * 100,
        0.36,
        color=orange,
    )
    range_axis.set_ylabel("sampled RGB out of range (%)", color=orange)
    axes[1, 1].set_title("Stability regularization works, with a fidelity cost")

    figure.suptitle(
        "Residual appearance crosses 20 dB with irregular geometry bitwise frozen",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plot_evidence(collect_evidence(Path(args.root)), Path(args.out))


if __name__ == "__main__":
    main()
