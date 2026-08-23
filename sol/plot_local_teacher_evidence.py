"""Plot matched teacher-distillation screens and exact-render evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARM_ORDER = ("control", "appearance", "full")
AUDIT_ARMS = {
    "control": "irregular_seed0",
    "appearance": "irregular_seed1",
    "full": "irregular_seed2",
}


def _label_bars(axis, bars, *, precision: int) -> None:
    """Place exact values above bars without changing the evidence scale."""
    for bar in bars:
        axis.annotate(
            f"{bar.get_height():.{precision}f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )


def load_screen_metrics(root: Path) -> dict[str, dict[str, float]]:
    """Load the final held-out sampled-render metrics for each matched arm."""
    metrics = {}
    for arm in ARM_ORDER:
        summary = json.loads(
            (root / f"{arm}_seed0_600" / "summary.json").read_text()
        )["latest_evaluation"]
        structure = summary["structure"]
        metrics[arm] = {
            "psnr": float(summary["macro_psnr"]),
            "occupancy": float(structure["occupancy_uniformity"]),
            "active": float(structure["active_fraction"]),
            "tilt": float(structure["mixed_spacetime_tilt_median"]),
            "extent": float(structure["extent_median"]),
        }
    return metrics


def load_exact_metrics(
    report_path: Path,
) -> tuple[dict[str, dict[str, float]], list[dict[str, float | str]]]:
    """Load exact macro metrics and per-style deltas relative to the control."""
    report = json.loads(report_path.read_text())
    macro = {
        arm: {
            "psnr": float(report["perceptual_macro"][audit_arm]["psnr"]),
            "lpips": float(report["perceptual_macro"][audit_arm]["lpips"]),
            "ssim": float(report["perceptual_macro"][audit_arm]["ssim"]),
        }
        for arm, audit_arm in AUDIT_ARMS.items()
    }

    records = {
        (record["style"], record["arm"]): record
        for record in report["perceptual_records"]
        if record["arm"] in AUDIT_ARMS.values()
    }
    styles = sorted({style for style, _ in records})
    deltas: list[dict[str, float | str]] = []
    for style in styles:
        control = records[(style, AUDIT_ARMS["control"])]
        for arm in ("appearance", "full"):
            candidate = records[(style, AUDIT_ARMS[arm])]
            deltas.append({
                "style": style,
                "arm": arm,
                "psnr_delta": float(
                    candidate["render_signature"]["psnr"]
                    - control["render_signature"]["psnr"]
                ),
                "lpips_improvement": float(
                    control["lpips_mean"] - candidate["lpips_mean"]
                ),
            })
    return macro, deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screens", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--title",
        default="Local teacher distillation: matched seed-0 continuation (600 steps)",
    )
    parser.add_argument("--appearance-label", default="appearance\nlocal")
    parser.add_argument("--full-label", default="full local")
    args = parser.parse_args()

    screens = load_screen_metrics(Path(args.screens))
    exact, style_deltas = load_exact_metrics(Path(args.audit))

    import matplotlib.pyplot as plt  # noqa: PLC0415

    labels = ["control", args.appearance_label, args.full_label]
    colors = ["#6b7280", "#2a6fbb", "#d97706"]
    positions = list(range(len(ARM_ORDER)))
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))

    sampled_psnr = [screens[arm]["psnr"] for arm in ARM_ORDER]
    sampled_bars = axes[0, 0].bar(positions, sampled_psnr, color=colors)
    _label_bars(axes[0, 0], sampled_bars, precision=3)
    axes[0, 0].axhspan(
        sampled_psnr[0] - 0.5, sampled_psnr[0], color="#6b7280", alpha=0.12
    )
    axes[0, 0].axhline(20.0, color="#a61b1b", linestyle="--", linewidth=1)
    axes[0, 0].set(
        title="Sampled-render screen", ylabel="PSNR (dB)",
        xticks=positions, xticklabels=labels,
        ylim=(min(sampled_psnr) - 0.35, 20.25),
    )

    exact_psnr = [exact[arm]["psnr"] for arm in ARM_ORDER]
    exact_psnr_bars = axes[0, 1].bar(positions, exact_psnr, color=colors)
    _label_bars(axes[0, 1], exact_psnr_bars, precision=3)
    axes[0, 1].axhline(20.0, color="#a61b1b", linestyle="--", linewidth=1)
    axes[0, 1].set(
        title="Exact-render fidelity", ylabel="PSNR (dB)",
        xticks=positions, xticklabels=labels,
        ylim=(min(exact_psnr) - 0.35, 20.25),
    )

    exact_lpips = [exact[arm]["lpips"] for arm in ARM_ORDER]
    exact_lpips_bars = axes[0, 2].bar(positions, exact_lpips, color=colors)
    _label_bars(axes[0, 2], exact_lpips_bars, precision=3)
    axes[0, 2].axhline(0.40, color="#a61b1b", linestyle="--", linewidth=1)
    axes[0, 2].set(
        title="Exact-render perception", ylabel="LPIPS (lower is better)",
        xticks=positions, xticklabels=labels,
        ylim=(0.35, max(exact_lpips) + 0.04),
    )

    occupancy = [screens[arm]["occupancy"] for arm in ARM_ORDER]
    occupancy_bars = axes[1, 0].bar(positions, occupancy, color=colors)
    _label_bars(axes[1, 0], occupancy_bars, precision=4)
    axes[1, 0].axhline(0.985, color="#a61b1b", linestyle="--", linewidth=1)
    axes[1, 0].set(
        title="Irregularity gate", ylabel="occupancy uniformity",
        xticks=positions, xticklabels=labels,
        ylim=(min(occupancy) - 0.004, 0.988),
    )

    width = 0.35
    active = [screens[arm]["active"] for arm in ARM_ORDER]
    tilt = [screens[arm]["tilt"] for arm in ARM_ORDER]
    axes[1, 1].bar(
        [position - width / 2 for position in positions], active,
        width=width, color="#5b8c5a", label="active fraction",
    )
    axes[1, 1].bar(
        [position + width / 2 for position in positions], tilt,
        width=width, color="#7b61a8", label="mixed tilt",
    )
    axes[1, 1].axhline(0.70, color="#a61b1b", linestyle="--", linewidth=1)
    axes[1, 1].axhline(0.25, color="#a61b1b", linestyle=":", linewidth=1)
    axes[1, 1].set(
        title="Sparsity and time distortion", ylabel="fraction",
        xticks=positions, xticklabels=labels, ylim=(0, 0.75),
    )
    axes[1, 1].legend(fontsize=8, loc="lower right")

    marker = {"appearance": "o", "full": "s"}
    arm_color = {"appearance": colors[1], "full": colors[2]}
    for arm in ("appearance", "full"):
        rows = [row for row in style_deltas if row["arm"] == arm]
        axes[1, 2].scatter(
            [row["psnr_delta"] for row in rows],
            [row["lpips_improvement"] for row in rows],
            color=arm_color[arm], marker=marker[arm], s=55, label=arm,
        )
        for row in rows:
            axes[1, 2].annotate(
                str(row["style"]),
                (float(row["psnr_delta"]), float(row["lpips_improvement"])),
                xytext=(4, 3), textcoords="offset points", fontsize=7,
            )
    axes[1, 2].axhline(0, color="#4b5563", linewidth=1)
    axes[1, 2].axvline(0, color="#4b5563", linewidth=1)
    axes[1, 2].set(
        title="Per-style exact improvement",
        xlabel="PSNR delta vs control (dB)",
        ylabel="LPIPS improvement vs control",
    )
    axes[1, 2].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle(args.title, fontsize=14)
    figure.tight_layout()
    figure.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
