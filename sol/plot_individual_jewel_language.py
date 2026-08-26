"""Plot the passing active individual-Jewel language gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(path: Path) -> dict:
    """Load only the registered Gate 0f metrics used by the evidence plot."""
    report = json.loads(path.read_text())
    if report.get("schema") != "active-individual-jewel-language-gate-v1":
        raise ValueError("report is not an active individual-Jewel language audit")
    macro = report["macro"]
    canonicality = report["canonicality"]["summary"]
    return {
        "passed": bool(report["gate"]["passed"]),
        "token_psnr": float(macro["token_only_voxel_psnr"]),
        "full_psnr": float(macro["full_residual_voxel_psnr"]),
        "tilt_retention": float(macro["token_only_mixed_tilt_retention"]),
        "center_lock_percent": 100.0
        * float(macro["token_only_cell_center_lock_fraction"]),
        "eight_frame_decisions": float(macro["eight_frame_decisions"]),
        "same_source": float(canonicality["same_source"]),
        "different_source": float(canonicality["different_source"]),
        "language_margin": float(canonicality["margin"]),
    }


def plot(metrics: dict, output: Path) -> None:
    """Render a compact four-panel pass/fail figure."""
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    axes[0].bar(["token only"], [metrics["token_psnr"]], color="tab:blue")
    axes[0].axhline(20, color="tab:red", linestyle="--", label="20 dB gate")
    axes[0].set_ylim(0, max(25.0, metrics["token_psnr"] * 1.12))
    axes[0].set(title="Renderable vocabulary", ylabel="random-volume PSNR (dB)")
    axes[0].legend()

    axes[1].bar(["token / source"], [metrics["tilt_retention"]], color="tab:green")
    axes[1].axhspan(0.85, 1.15, color="tab:green", alpha=0.12, label="registered band")
    axes[1].axhline(1, color="black", linewidth=1)
    axes[1].set_ylim(0.75, 1.22)
    axes[1].set(title="Spacetime structure", ylabel="mixed-tilt retention")
    axes[1].legend()

    names = ["same video", "different video", "margin"]
    values = [metrics["same_source"], metrics["different_source"], metrics["language_margin"]]
    axes[2].bar(names, values, color=["tab:blue", "tab:orange", "tab:green"])
    axes[2].axhline(0.05, color="tab:red", linestyle="--", label="margin gate")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].set(title="Independent-fit language", ylabel="cell-conditional cosine")
    axes[2].legend()

    decisions_k = metrics["eight_frame_decisions"] / 1000
    axes[3].bar(["decisions / 8 frames"], [decisions_k], color="tab:purple")
    axes[3].axhline(40, color="tab:red", linestyle="--", label="40k gate")
    axes[3].set_ylim(0, 45)
    axes[3].text(0, decisions_k / 2, f"grid lock\n{metrics['center_lock_percent']:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    axes[3].set(title="Irregular generation budget", ylabel="role decisions (thousands)")
    axes[3].legend()
    verdict = "PASS" if metrics["passed"] else "FAIL"
    figure.suptitle(f"Active individual-Jewel language Gate 0f — {verdict}", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    metrics = load_metrics(Path(args.report))
    plot(metrics, Path(args.out))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
