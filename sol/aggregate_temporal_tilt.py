"""Aggregate multi-source temporal-tilt ablations into a replication gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from sol.temporal_tilt_ablation import mean_confidence_interval, write_report


def _source_label(source: str) -> str:
    """Return a stable human-readable source label."""
    return "synthetic_tube" if source == "synthetic_tube" else Path(source).stem


def aggregate_reports(reports: list[dict]) -> dict:
    """Combine v2 per-source reports and evaluate the decision-grade gate."""
    if not reports:
        raise ValueError("at least one report is required")
    pairs = []
    protocol = reports[0].get("protocol")
    for report in reports:
        if report.get("schema") != "temporal-tilt-ablation-v2":
            raise ValueError("all inputs must use temporal-tilt-ablation-v2")
        if report.get("protocol") != protocol:
            raise ValueError("all input reports must use the same protocol")
        source = report["source"]
        largest_steps = report["summary"]["largest_budget_axis_aligned"]["steps"]
        for comparison in report["summary"]["comparisons"]:
            if (
                comparison["control"] == "axis_aligned"
                and comparison["steps"] == largest_steps
            ):
                pairs.append(
                    {
                        "source": source,
                        "source_label": _source_label(source),
                        **comparison,
                    }
                )
    if not pairs:
        raise ValueError("no complete free/axis-aligned pairs found")

    delta = [item["free_minus_control_psnr_db"] for item in pairs]
    rgb_error = [item["control_minus_free_rgb_mae"] for item in pairs]
    motion_error = [
        item["control_minus_free_motion_top20_rgb_mae"] for item in pairs
    ]
    sources = sorted({item["source"] for item in pairs})
    per_source = {}
    for source in sources:
        source_pairs = [item for item in pairs if item["source"] == source]
        per_source[_source_label(source)] = {
            "source": source,
            "pair_count": len(source_pairs),
            "free_wins": sum(
                item["free_minus_control_psnr_db"] > 0 for item in source_pairs
            ),
            "paired_psnr_delta_db": mean_confidence_interval(
                [item["free_minus_control_psnr_db"] for item in source_pairs]
            ),
            "control_minus_free_rgb_mae": mean_confidence_interval(
                [item["control_minus_free_rgb_mae"] for item in source_pairs]
            ),
            "control_minus_free_motion_top20_rgb_mae": (
                mean_confidence_interval(
                    [
                        item["control_minus_free_motion_top20_rgb_mae"]
                        for item in source_pairs
                    ]
                )
            ),
        }

    psnr_stats = mean_confidence_interval(delta)
    free_wins = sum(value > 0 for value in delta)
    matched_counts = all(item["free_minus_control_primitives"] == 0 for item in pairs)
    matched_bytes = all(
        item.get("free_parameter_bytes") == item.get("control_parameter_bytes")
        for item in pairs
    )
    controls_projected = all(
        item["control_mixed_tilt_median"] <= 1e-5 for item in pairs
    )
    return {
        "schema": "temporal-tilt-replication-v1",
        "protocol": protocol,
        "sources": [
            {
                "source": report["source"],
                "source_label": _source_label(report["source"]),
                "source_fingerprint": report.get("source_fingerprint"),
                "environment": report.get("environment"),
            }
            for report in reports
        ],
        "source_count": len(sources),
        "pair_count": len(pairs),
        "pairs": pairs,
        "per_source": per_source,
        "aggregate": {
            "free_wins": free_wins,
            "paired_psnr_delta_db": psnr_stats,
            "control_minus_free_rgb_mae": mean_confidence_interval(rgb_error),
            "control_minus_free_motion_top20_rgb_mae": (
                mean_confidence_interval(motion_error)
            ),
            "free_mixed_tilt_median_mean": statistics.mean(
                item["free_mixed_tilt_median"] for item in pairs
            ),
            "free_psnr_db_per_1000_primitives_mean": statistics.mean(
                item["free_psnr_db_per_1000_primitives"] for item in pairs
            ),
            "control_psnr_db_per_1000_primitives_mean": statistics.mean(
                item["control_psnr_db_per_1000_primitives"] for item in pairs
            ),
            "free_psnr_db_per_parameter_megabyte_mean": statistics.mean(
                item["free_psnr_db_per_parameter_megabyte"] for item in pairs
            ),
            "control_psnr_db_per_parameter_megabyte_mean": statistics.mean(
                item["control_psnr_db_per_parameter_megabyte"] for item in pairs
            ),
            "primitive_counts_matched": matched_counts,
            "parameter_bytes_matched": matched_bytes,
        },
        "decision_gate": {
            "at_least_three_sources": len(sources) >= 3,
            "at_least_nine_pairs": len(pairs) >= 9,
            "free_wins_at_least_seven_pairs": free_wins >= 7,
            "macro_advantage_at_least_0_5db": bool(
                psnr_stats["mean"] is not None and psnr_stats["mean"] >= 0.5
            ),
            "paired_ci95_excludes_zero": bool(
                psnr_stats["ci95_low"] is not None
                and psnr_stats["ci95_low"] > 0
            ),
            "primitive_counts_matched": matched_counts,
            "parameter_bytes_matched": matched_bytes,
            "projection_removes_mixed_tilt": controls_projected,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text()) for path in args.reports]
    aggregate = aggregate_reports(reports)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_report(output, aggregate)
    stats = aggregate["aggregate"]["paired_psnr_delta_db"]
    print(
        f"{aggregate['pair_count']} pairs across {aggregate['source_count']} sources: "
        f"free-axis delta {stats['mean']:.3f} dB "
        f"95% CI [{stats['ci95_low']:.3f}, {stats['ci95_high']:.3f}]"
    )


if __name__ == "__main__":
    main()
