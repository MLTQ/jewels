"""Plot the bounded-to-residual appearance-contract experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARM_ORDER = (
    "bounded_control",
    "bounded_midpoint",
    "residual_control",
    "raw_response",
)
AUDIT_ARMS = {
    "bounded_control": "irregular_seed0",
    "bounded_midpoint": "irregular_seed1",
    "residual_control": "irregular_seed0",
    "raw_response": "irregular_seed1",
}


def _summary(path: Path) -> dict[str, float]:
    final = json.loads(path.read_text())["latest_evaluation"]
    structure = final["structure"]
    return {
        "sampled_psnr": float(final["macro_psnr"]),
        "occupancy": float(structure["occupancy_uniformity"]),
        "active": float(structure["active_fraction"]),
        "tilt": float(structure["mixed_spacetime_tilt_median"]),
    }


def _exact(report: dict, audit_arm: str) -> dict[str, float]:
    row = report["perceptual_macro"][audit_arm]
    return {
        "psnr": float(row["psnr"]),
        "lpips": float(row["lpips"]),
        "ssim": float(row["ssim"]),
        "layout_psnr": float(row["layout_psnr"]),
    }


def load_evidence(
    bounded_screens: Path,
    bounded_audit: Path,
    residual_screens: Path,
    residual_audit: Path,
) -> tuple[dict[str, dict[str, float]], list[dict[str, float | str]]]:
    """Load four arm metrics and response-vs-control per-style exact deltas."""
    bounded_report = json.loads(bounded_audit.read_text())
    residual_report = json.loads(residual_audit.read_text())
    screen_paths = {
        "bounded_control": bounded_screens / "control_seed0_600" / "summary.json",
        "bounded_midpoint": (
            bounded_screens / "hybrid_midpoint_seed0_600" / "summary.json"
        ),
        "residual_control": residual_screens / "control_seed0_600" / "summary.json",
        "raw_response": residual_screens / "response_seed0_600" / "summary.json",
    }
    reports = {
        "bounded_control": bounded_report,
        "bounded_midpoint": bounded_report,
        "residual_control": residual_report,
        "raw_response": residual_report,
    }
    metrics = {
        arm: {
            **_summary(screen_paths[arm]),
            **_exact(reports[arm], AUDIT_ARMS[arm]),
        }
        for arm in ARM_ORDER
    }

    records = {
        (record["style"], record["arm"]): record
        for record in residual_report["perceptual_records"]
        if record["arm"] in ("irregular_seed0", "irregular_seed1")
    }
    deltas: list[dict[str, float | str]] = []
    for style in sorted({style for style, _ in records}):
        control = records[(style, "irregular_seed0")]
        response = records[(style, "irregular_seed1")]
        deltas.append({
            "style": style,
            "psnr_delta": float(
                response["render_signature"]["psnr"]
                - control["render_signature"]["psnr"]
            ),
            "lpips_improvement": float(
                control["lpips_mean"] - response["lpips_mean"]
            ),
        })
    return metrics, deltas


def _label_bars(axis, bars, precision: int) -> None:
    for bar in bars:
        axis.annotate(
            f"{bar.get_height():.{precision}f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounded-screens", required=True)
    parser.add_argument("--bounded-audit", required=True)
    parser.add_argument("--residual-screens", required=True)
    parser.add_argument("--residual-audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    metrics, style_deltas = load_evidence(
        Path(args.bounded_screens), Path(args.bounded_audit),
        Path(args.residual_screens), Path(args.residual_audit),
    )

    import matplotlib.pyplot as plt  # noqa: PLC0415

    labels = ["bounded\ncontrol", "bounded\nmidpoint", "residual\ncontrol", "raw\nresponse"]
    colors = ["#6b7280", "#a78b6d", "#2a6fbb", "#d97706"]
    positions = list(range(len(ARM_ORDER)))
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))

    panels = (
        (axes[0, 0], "sampled_psnr", "Sampled-render screen", "PSNR (dB)", 20.0, 3),
        (axes[0, 1], "psnr", "Exact-render fidelity", "PSNR (dB)", 20.0, 3),
        (axes[0, 2], "lpips", "Exact-render perception", "LPIPS (lower is better)", 0.40, 3),
        (axes[1, 0], "ssim", "Exact structural similarity", "SSIM", None, 3),
    )
    for axis, key, title, ylabel, threshold, precision in panels:
        values = [metrics[arm][key] for arm in ARM_ORDER]
        bars = axis.bar(positions, values, color=colors)
        _label_bars(axis, bars, precision)
        if threshold is not None:
            axis.axhline(threshold, color="#a61b1b", linestyle="--", linewidth=1)
        margin = max(max(values) - min(values), 0.02)
        lower = max(0.0, min(values) - 0.35 * margin)
        upper = max(values) + 0.35 * margin
        if key == "lpips":
            lower = min(0.35, lower)
        if threshold is not None:
            lower = min(lower, threshold - 0.03)
            upper = max(upper, threshold + 0.03)
        axis.set(
            title=title, ylabel=ylabel,
            xticks=positions, xticklabels=labels, ylim=(lower, upper),
        )

    residual_positions = [0, 1]
    residual_arms = ("residual_control", "raw_response")
    width = 0.24
    structure = (
        ("occupancy", -width, "#8c564b", "occupancy"),
        ("active", 0.0, "#5b8c5a", "active"),
        ("tilt", width, "#7b61a8", "mixed tilt"),
    )
    for key, offset, color, label in structure:
        axes[1, 1].bar(
            [position + offset for position in residual_positions],
            [metrics[arm][key] for arm in residual_arms],
            width=width, color=color, label=label,
        )
    axes[1, 1].axhline(0.985, color="#a61b1b", linestyle="--", linewidth=1)
    axes[1, 1].axhline(0.70, color="#a61b1b", linestyle="-.", linewidth=1)
    axes[1, 1].axhline(0.25, color="#a61b1b", linestyle=":", linewidth=1)
    axes[1, 1].set(
        title="Residual structure gates", ylabel="metric",
        xticks=residual_positions, xticklabels=["control", "raw response"],
        ylim=(0, 1.02),
    )
    axes[1, 1].legend(fontsize=8, loc="lower right")

    axes[1, 2].scatter(
        [row["psnr_delta"] for row in style_deltas],
        [row["lpips_improvement"] for row in style_deltas],
        color=colors[-1], s=60,
    )
    for row in style_deltas:
        axes[1, 2].annotate(
            str(row["style"]),
            (float(row["psnr_delta"]), float(row["lpips_improvement"])),
            xytext=(4, 3), textcoords="offset points", fontsize=8,
        )
    axes[1, 2].axhline(0, color="#4b5563", linewidth=1)
    axes[1, 2].axvline(0, color="#4b5563", linewidth=1)
    axes[1, 2].set(
        title="Raw response vs residual control by style",
        xlabel="exact PSNR delta (dB)", ylabel="LPIPS improvement",
    )

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle(
        "Residual appearance contract: matched seed-0 continuation (600 steps)",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
