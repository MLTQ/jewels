"""Aggregate the preregistered bundle-size curve for Jewel casting Gate 0c."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_granularity_gate(reports: dict[int, dict]) -> dict:
    """Extract the matched curve and evaluate the frozen bundle-1 upper-bound gate."""
    expected = [8, 4, 2, 1]
    if sorted(reports, reverse=True) != expected:
        raise ValueError("granularity gate requires bundle sizes 8, 4, 2, and 1")
    curve = []
    for bundle_size in expected:
        report = reports[bundle_size]
        if report.get("schema") != "factorized-jewel-casting-language-gate-v1":
            raise ValueError("granularity input has an unsupported schema")
        if int(report["protocol"]["bundle_size"]) != bundle_size:
            raise ValueError("granularity report is assigned to the wrong bundle size")
        selected = report["vocabularies"]["1024"]
        macro = selected["macro"]
        serialized = all(
            row["source_jewels"]
            == row["program"]["source_jewels"]
            == row["program"]["serialized_jewels"]
            for row in selected["records"]
        )
        curve.append(
            {
                "bundle_size": bundle_size,
                "serialized": serialized,
                **macro,
                "canonical_margin": selected["canonicality"]["summary"][
                    "composite_cell_conditional_cosine"
                ]["margin"],
                "eight_frame_discrete_decisions": macro["discrete_decisions"] * 8 / 49,
            }
        )
    primary = curve[-1]
    baseline = curve[0]
    token_curve = [row["token_only_voxel_psnr"] for row in curve]
    half_curve = [row["half_residual_voxel_psnr"] for row in curve]
    checks = {
        "all_jewels_serialized": all(row["serialized"] for row in curve),
        "full_residual_numerically_exact": primary["full_residual_voxel_psnr"] >= 80,
        "token_centers_not_grid_locked": primary["token_only_cell_center_lock_fraction"] < 0.01,
        "token_only_at_least_20db": primary["token_only_voxel_psnr"] >= 20,
        "token_only_improves_bundle8_by_3db": (
            primary["token_only_voxel_psnr"] - baseline["token_only_voxel_psnr"] >= 3
        ),
        "half_residual_at_least_25db": primary["half_residual_voxel_psnr"] >= 25,
        "half_residual_tilt_within_10pct": (
            0.9 <= primary["half_residual_mixed_tilt_retention"] <= 1.1
        ),
        "same_source_margin_at_least_002": primary["canonical_margin"] >= 0.02,
        "eight_frame_decisions_at_most_50000": (
            primary["eight_frame_discrete_decisions"] <= 50000
        ),
        "token_psnr_monotonic": all(
            second >= first for first, second in zip(token_curve, token_curve[1:])
        ),
        "half_residual_psnr_monotonic": all(
            second >= first for first, second in zip(half_curve, half_curve[1:])
        ),
    }
    return {
        "schema": "jewel-casting-granularity-gate-v1",
        "curve": curve,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for bundle_size in (8, 4, 2, 1):
        parser.add_argument(f"--bundle{bundle_size}", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    reports = {
        bundle_size: json.loads(
            Path(getattr(args, f"bundle{bundle_size}")).read_text()
        )
        for bundle_size in (8, 4, 2, 1)
    }
    result = evaluate_granularity_gate(reports)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")

    import matplotlib.pyplot as plt  # noqa: PLC0415

    curve = result["curve"]
    sizes = [row["bundle_size"] for row in curve]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.1))
    axes[0].plot(
        sizes, [100 * row["motif_explained_fraction"] for row in curve], marker="o"
    )
    axes[0].set(title="Compositional capacity", ylabel="bundle energy explained (%)")
    for name in ("covariance", "surface", "gradient"):
        axes[0].plot(
            sizes,
            [100 * row["factor_explained_fraction"][name] for row in curve],
            marker=".", linestyle="--", label=name,
        )
    axes[0].legend(fontsize=7)

    axes[1].plot(
        sizes, [row["token_only_voxel_psnr"] for row in curve],
        marker="o", label="tokens only",
    )
    axes[1].plot(
        sizes, [row["half_residual_voxel_psnr"] for row in curve],
        marker="o", label="tokens + 50% residual",
    )
    axes[1].axhline(20, color="tab:red", linestyle="--", linewidth=1)
    axes[1].axhline(25, color="tab:red", linestyle=":", linewidth=1)
    axes[1].set(title="Rendering vs cast granularity", ylabel="random-volume PSNR (dB)")
    axes[1].legend(fontsize=7)

    axes[2].plot(
        sizes, [row["canonical_margin"] for row in curve], marker="o"
    )
    axes[2].axhline(0.02, color="tab:red", linestyle="--", linewidth=1)
    axes[2].set(title="Independent-fit micro language", ylabel="same − different cosine")

    axes[3].plot(
        sizes, [row["eight_frame_discrete_decisions"] for row in curve], marker="o"
    )
    axes[3].axhline(50000, color="tab:red", linestyle="--", linewidth=1)
    axes[3].set(title="Generation cost", ylabel="role decisions / 8 frames")

    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.invert_xaxis()
        axis.set_xticks(sizes, labels=sizes)
        axis.set_xlabel("Jewels per cast")
        axis.grid(alpha=0.22)
    verdict = "PASS" if result["gate"]["passed"] else "FAIL"
    figure.suptitle(f"Jewel casting granularity Gate 0c — {verdict}", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output / "granularity_curve.png", dpi=200)
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__":
    main()
