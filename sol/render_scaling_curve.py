"""Render the fitted-data scaling curve from committed evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect_point(
    sources: int, mark_summary: dict, perceptual_report: dict, arm: str
) -> dict:
    """Extract one curve point from a mark-flow summary and a perceptual report."""
    if sources <= 0:
        raise ValueError("sources must be positive")
    evaluation = mark_summary["latest_evaluation"]
    macro = perceptual_report["macro"]
    if arm not in macro:
        raise ValueError(f"perceptual report lacks arm {arm!r}")
    correct = evaluation["aggregate"]["correct"]
    shuffled = evaluation["aggregate"].get("shuffled_scaffold")
    return {
        "sources": sources,
        "correct_feature_loss": float(correct),
        "shuffled_minus_correct": (
            float(shuffled) - float(correct) if shuffled is not None else None
        ),
        "rollout_psnr": float(macro[arm]["psnr"]),
        "rollout_ssim": float(macro[arm]["ssim"]),
        "rollout_lpips": float(macro[arm]["lpips_mean"]),
    }


def render_curve(points: list[dict], out_path: Path) -> None:
    """Draw one small-multiple panel per metric over fitted training sources."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(points) < 2:
        raise ValueError("a curve needs at least two points")
    ordered = sorted(points, key=lambda point: point["sources"])
    x = [point["sources"] for point in ordered]
    panels = (
        ("correct_feature_loss", "held-out velocity loss", True),
        ("rollout_psnr", "rollout PSNR (dB)", False),
        ("rollout_lpips", "rollout LPIPS", True),
    )
    hue = "#3b6fb6"
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharex=True)
    for axis, (key, label, lower_better) in zip(axes, panels):
        y = [point[key] for point in ordered]
        axis.plot(x, y, color=hue, linewidth=2, marker="o", markersize=7)
        for px, py in zip(x, y):
            axis.annotate(
                f"{py:.3f}" if abs(py) < 10 else f"{py:.2f}",
                (px, py),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color="#333333",
            )
        axis.set_title(
            f"{label} ({'lower' if lower_better else 'higher'} is better)",
            fontsize=9,
        )
        axis.set_xscale("log", base=2)
        axis.set_xticks(x)
        axis.set_xticklabels([str(value) for value in x])
        axis.grid(True, color="#dddddd", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(y=0.22)
    axes[0].set_xlabel("")
    axes[1].set_xlabel("fitted training sources (class-balanced)", fontsize=9)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--point",
        action="append",
        required=True,
        help="sources=N,mark=<summary.json>,report=<report.json>,arm=<macro arm>",
    )
    parser.add_argument("--out-figure", required=True)
    parser.add_argument("--out-table", required=True)
    args = parser.parse_args()
    points = []
    for entry in args.point:
        fields = dict(part.split("=", 1) for part in entry.split(","))
        missing = {"sources", "mark", "report", "arm"} - set(fields)
        if missing:
            raise ValueError(f"--point missing fields: {sorted(missing)}")
        points.append(
            collect_point(
                int(fields["sources"]),
                json.loads(Path(fields["mark"]).read_text()),
                json.loads(Path(fields["report"]).read_text()),
                fields["arm"],
            )
        )
    render_curve(points, Path(args.out_figure))
    table = {"schema": "realizer-data-scaling-v1", "points": points}
    Path(args.out_table).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_table).write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
