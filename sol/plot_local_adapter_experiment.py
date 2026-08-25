"""Plot the frozen-base local appearance adapter experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _metric(root: Path, report: str, key: str) -> dict[str, float]:
    saved = json.loads((root / report / "report.json").read_text())
    return saved["perceptual_macro"][key]


def _last_training_record(path: Path) -> dict[str, float]:
    records = [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]
    training = [row for row in records if "render_out_of_range_fraction" in row]
    if not training:
        raise ValueError(f"no training diagnostics in {path}")
    return training[-1]


def collect_evidence(root: Path) -> dict:
    """Collect semantically labeled exact metrics and final train diagnostics."""
    arms = {
        "frozen source": _metric(root, "audit_final_seed0_400", "irregular_seed0"),
        "radius-0 render": _metric(root, "audit_seed0_400", "irregular_seed1"),
        "radius-2 render": _metric(root, "audit_seed0_400", "irregular_seed2"),
        "radius-0 LPIPS .01": _metric(
            root, "audit_lpips_strength_seed0_400", "irregular_seed1"
        ),
        "radius-2 LPIPS .01": _metric(
            root, "audit_lpips_strength_seed0_400", "irregular_seed2"
        ),
        "raw local LPIPS .05": _metric(
            root, "audit_final_seed0_400", "irregular_seed2"
        ),
        "derivative x1 LPIPS .05": _metric(
            root, "audit_derivative_seed0_400", "irregular_seed3"
        ),
        "derivative x32 LPIPS .05": _metric(
            root, "audit_final_seed0_400", "irregular_seed3"
        ),
    }
    diagnostics = {}
    log_dirs = {
        "radius-2 render": "radius2_render_seed0_400",
        "raw LPIPS .01": "radius2_lpips001_seed0_400",
        "raw LPIPS .05": "radius2_lpips005_seed0_400",
        "derivative x32": "derivative_scale32_lpips005_seed0_400",
    }
    for label, directory in log_dirs.items():
        row = _last_training_record(root / "screens" / directory / "train_log.jsonl")
        diagnostics[label] = {
            "out_of_range_percent": 100.0 * row["render_out_of_range_fraction"],
            "residual_gradient_energy": row["residual_gradient_energy"],
        }
    causal = {
        "render": {
            "psnr_millidb": 1000.0 * (
                arms["radius-2 render"]["psnr"] - arms["radius-0 render"]["psnr"]
            ),
            "lpips_improvement_milli": 1000.0 * (
                arms["radius-0 render"]["lpips"] - arms["radius-2 render"]["lpips"]
            ),
        },
        "LPIPS .01": {
            "psnr_millidb": 1000.0 * (
                arms["radius-2 LPIPS .01"]["psnr"]
                - arms["radius-0 LPIPS .01"]["psnr"]
            ),
            "lpips_improvement_milli": 1000.0 * (
                arms["radius-0 LPIPS .01"]["lpips"]
                - arms["radius-2 LPIPS .01"]["lpips"]
            ),
        },
    }
    return {"arms": arms, "causal_radius2_minus_radius0": causal, "diagnostics": diagnostics}


def plot_evidence(evidence: dict, out: Path) -> None:
    """Render the exact frontier, causal controls, and stability diagnostics."""
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    arms = evidence["arms"]

    frontier_names = (
        "frozen source",
        "radius-0 LPIPS .01",
        "raw local LPIPS .05",
        "derivative x32 LPIPS .05",
    )
    colors = ("#4c566a", "#5e81ac", "#bf616a", "#a3be8c")
    annotation_offsets = ((-35, 8), (5, 8), (-92, -13), (5, 5))
    for name, color, offset in zip(
        frontier_names, colors, annotation_offsets, strict=True
    ):
        row = arms[name]
        axes[0, 0].scatter(row["psnr"], row["lpips"], s=90, color=color)
        axes[0, 0].annotate(name, (row["psnr"], row["lpips"]), xytext=offset,
                            textcoords="offset points", fontsize=9)
    axes[0, 0].axvline(20.0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].axhline(0.70, color="#d08770", linestyle="--", linewidth=1.5)
    axes[0, 0].set(xlabel="Exact PSNR (dB; right is better)",
                   ylabel="Exact LPIPS (lower is better)",
                   title="All adapters preserve 20 dB; none reaches LPIPS < 0.70")

    causal = evidence["causal_radius2_minus_radius0"]
    labels = list(causal)
    x = range(len(labels))
    width = 0.35
    axes[0, 1].bar(
        [value - width / 2 for value in x],
        [causal[label]["psnr_millidb"] for label in labels],
        width, label="PSNR gain (milli-dB)", color="#5e81ac",
    )
    axes[0, 1].bar(
        [value + width / 2 for value in x],
        [causal[label]["lpips_improvement_milli"] for label in labels],
        width, label="LPIPS improvement (x1000)", color="#d08770",
    )
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].set_xticks(list(x), labels)
    axes[0, 1].set_title("Radius-2 minus radius-0: all effects < 0.00004")
    axes[0, 1].legend(fontsize=8)

    sequence = (
        "frozen source",
        "radius-2 LPIPS .01",
        "raw local LPIPS .05",
        "derivative x1 LPIPS .05",
        "derivative x32 LPIPS .05",
    )
    short = ("source", "raw .01", "raw .05", "deriv x1", "deriv x32")
    lpips_values = [arms[name]["lpips"] for name in sequence]
    psnr_values = [arms[name]["psnr"] for name in sequence]
    axis_lpips = axes[1, 0]
    axis_psnr = axis_lpips.twinx()
    axis_lpips.plot(short, lpips_values, marker="o", color="#bf616a", label="LPIPS")
    axis_psnr.plot(short, psnr_values, marker="s", color="#5e81ac", label="PSNR")
    axis_lpips.axhline(0.70, color="#bf616a", linestyle="--", linewidth=1)
    axis_psnr.axhline(20.0, color="#5e81ac", linestyle="--", linewidth=1)
    axis_lpips.set_ylabel("LPIPS", color="#bf616a")
    axis_psnr.set_ylabel("PSNR (dB)", color="#5e81ac")
    axis_lpips.set_title("Perceptual pressure helps; forced evidence trades LPIPS for headroom")
    axis_lpips.tick_params(axis="x", rotation=18)

    diagnostics = evidence["diagnostics"]
    labels = list(diagnostics)
    x = list(range(len(labels)))
    axis_range = axes[1, 1]
    axis_energy = axis_range.twinx()
    axis_range.bar(
        [value - width / 2 for value in x],
        [diagnostics[label]["out_of_range_percent"] for label in labels],
        width, color="#88c0d0", label="out of range (%)",
    )
    axis_energy.bar(
        [value + width / 2 for value in x],
        [diagnostics[label]["residual_gradient_energy"] for label in labels],
        width, color="#b48ead", label="Jacobian energy",
    )
    axis_range.set_xticks(x, labels, rotation=18)
    axis_range.set_ylabel("Sampled RGB outside [0,1] (%)", color="#5e81ac")
    axis_energy.set_ylabel("Residual Jacobian energy", color="#b48ead")
    axis_range.set_title("Forced derivatives retain range but spend Jacobian energy")
    axis_range.legend(loc="upper left", fontsize=8)
    axis_energy.legend(loc="upper right", fontsize=8)

    figure.suptitle(
        "Frozen irregular Jewelfield: native local appearance adapter, seed 0",
        fontsize=16,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    evidence = collect_evidence(root)
    plot_evidence(evidence, out)
    out.with_suffix(".json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
