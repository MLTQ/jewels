"""Plot the fresh hierarchical Jewel-language result against its granularity curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def hierarchical_plot_payload(hierarchical: dict, granularity: dict) -> dict:
    """Validate and combine the fresh hierarchy with its registered precursor curve."""
    if hierarchical.get("schema") != "hierarchical-jewel-casting-language-gate-v1":
        raise ValueError("unsupported hierarchical report schema")
    if granularity.get("schema") != "jewel-casting-granularity-gate-v1":
        raise ValueError("unsupported granularity report schema")
    curve = granularity["curve"]
    labels = [f"bundle {row['bundle_size']}" for row in curve] + ["hierarchy\n(fresh)"]
    macro = hierarchical["macro"]
    return {
        "labels": labels,
        "token_psnr": [row["token_only_voxel_psnr"] for row in curve]
        + [macro["token_only_voxel_psnr"]],
        "half_psnr": [row["half_residual_voxel_psnr"] for row in curve]
        + [macro["half_residual_voxel_psnr"]],
        "half_tilt": [row["half_residual_mixed_tilt_retention"] for row in curve]
        + [macro["half_residual_mixed_tilt_retention"]],
        "canonical_margin": [row["canonical_margin"] for row in curve]
        + [hierarchical["canonicality"]["summary"]["margin"]],
        "eight_frame_decisions": [row["eight_frame_discrete_decisions"] for row in curve]
        + [macro["eight_frame_decisions"]],
        "gate_passed": bool(hierarchical["gate"]["passed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchical-report", required=True)
    parser.add_argument("--granularity-report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = hierarchical_plot_payload(
        json.loads(Path(args.hierarchical_report).read_text()),
        json.loads(Path(args.granularity_report).read_text()),
    )

    import matplotlib.pyplot as plt  # noqa: PLC0415

    x = list(range(len(payload["labels"])))
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].plot(x, payload["token_psnr"], marker="o", label="tokens only")
    axes[0].plot(x, payload["half_psnr"], marker="o", label="+ 50% residual")
    axes[0].axhline(20, color="tab:red", linestyle="--", linewidth=1)
    axes[0].axhline(25, color="tab:red", linestyle=":", linewidth=1)
    axes[0].set(title="Render-capable language", ylabel="random-volume PSNR (dB)")
    axes[0].legend(fontsize=8)

    axes[1].plot(x, payload["half_tilt"], marker="o")
    axes[1].axhspan(0.9, 1.1, color="tab:green", alpha=0.13)
    axes[1].axhline(1, color="black", linewidth=1)
    axes[1].set(title="Spacetime structure", ylabel="half-residual tilt / source")

    axes[2].plot(x, payload["canonical_margin"], marker="o")
    axes[2].axhline(0.05, color="tab:red", linestyle="--", linewidth=1)
    axes[2].set(title="Independent-fit language", ylabel="same − different cosine")

    axes[3].plot(x, payload["eight_frame_decisions"], marker="o")
    axes[3].axhline(40000, color="tab:red", linestyle="--", linewidth=1)
    axes[3].set(title="Generation cost", ylabel="role decisions / 8 frames")

    for axis in axes:
        axis.set_xticks(x, labels=payload["labels"], rotation=30, ha="right")
        axis.grid(alpha=0.22)
    verdict = "PASS" if payload["gate_passed"] else "FAIL"
    figure.suptitle(
        f"Hierarchical Jewel casting language Gate 0d — {verdict}",
        fontweight="bold",
    )
    figure.tight_layout()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)


if __name__ == "__main__":
    main()
