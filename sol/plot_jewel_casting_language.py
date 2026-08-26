"""Plot the preregistered Jewel casting-language gate as a pitch-readable figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_payload(report: dict) -> dict[str, list[float] | bool]:
    """Validate the report schema and extract vocabulary-ordered plotting series."""
    if report.get("schema") != "jewel-casting-language-gate-v0":
        raise ValueError("unsupported Jewel casting-language report schema")
    vocabularies = sorted(int(key) for key in report["vocabularies"])
    if not vocabularies:
        raise ValueError("casting-language report has no vocabularies")
    rows = [report["vocabularies"][str(size)] for size in vocabularies]

    def macro(key: str) -> list[float]:
        return [float(row["macro"][key]) for row in rows]

    def canonical(group: str) -> list[float]:
        return [
            float(
                row["canonicality"]["summary"][
                    "cell_conditional_motif_cosine"
                ][group]
            )
            for row in rows
        ]

    return {
        "vocabularies": vocabularies,
        "motif_explained_fraction": macro("motif_explained_fraction"),
        "token_only_psnr": macro("token_only_voxel_psnr"),
        "half_residual_psnr": macro("half_residual_voxel_psnr"),
        "grid_control_psnr": macro("grid_control_voxel_psnr"),
        "same_source_cosine": canonical("same_source"),
        "different_source_cosine": canonical("different_source"),
        "canonical_margin": canonical("margin"),
        "token_center_lock": macro("token_only_cell_center_lock_fraction"),
        "grid_center_lock": macro("grid_control_cell_center_lock_fraction"),
        "gate_passed": bool(report.get("gate", {}).get("passed", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    data = plot_payload(report)
    sizes = data["vocabularies"]

    import matplotlib.pyplot as plt  # noqa: PLC0415

    figure, axes = plt.subplots(1, 4, figsize=(16, 4.1))
    axes[0].plot(
        sizes, [100 * value for value in data["motif_explained_fraction"]],
        marker="o", linewidth=2,
    )
    axes[0].axhline(25, color="tab:red", linestyle="--", linewidth=1)
    axes[0].set(title="Motif carries structure", ylabel="bundle energy explained (%)")

    axes[1].plot(sizes, data["token_only_psnr"], marker="o", label="motif only")
    axes[1].plot(sizes, data["half_residual_psnr"], marker="o", label="motif + 50% residual")
    axes[1].plot(sizes, data["grid_control_psnr"], marker="x", label="grid-center control")
    axes[1].axhline(30, color="tab:red", linestyle="--", linewidth=1)
    axes[1].set(title="Continuous recovery", ylabel="random-volume PSNR (dB)")
    axes[1].legend(fontsize=8)

    axes[2].plot(sizes, data["same_source_cosine"], marker="o", label="same video")
    axes[2].plot(
        sizes, data["different_source_cosine"], marker="o", label="different video"
    )
    axes[2].plot(sizes, data["canonical_margin"], marker="s", label="margin")
    axes[2].axhline(0.05, color="tab:red", linestyle="--", linewidth=1)
    axes[2].set(title="Independent-fit language", ylabel="cell-conditional cosine")
    axes[2].legend(fontsize=8)

    axes[3].plot(
        sizes, [100 * value for value in data["token_center_lock"]],
        marker="o", label="motif casts",
    )
    axes[3].plot(
        sizes, [100 * value for value in data["grid_center_lock"]],
        marker="x", label="grid control",
    )
    axes[3].axhline(1, color="tab:red", linestyle="--", linewidth=1)
    axes[3].set(title="No lattice reintroduced", ylabel="exact center locking (%)")
    axes[3].legend(fontsize=8)

    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(sizes, labels=sizes)
        axis.set_xlabel("motif vocabulary")
        axis.grid(alpha=0.22)
    verdict = "PASS" if data["gate_passed"] else "FAIL"
    figure.suptitle(f"Jewel casting language Gate 0 — {verdict}", fontweight="bold")
    figure.tight_layout()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)


if __name__ == "__main__":
    main()
