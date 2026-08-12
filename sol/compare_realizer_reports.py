"""Compare matched video-to-jewel visual reports without metric cherry-picking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


RENDER_METRICS = (
    "psnr",
    "ssim",
    "contrast_ratio",
    "edge_ratio",
    "saturation_ratio",
    "temporal_change_ratio",
)
TOPOLOGY_METRICS = (
    "spatial_cell_fraction",
    "birth_cell_fraction",
    "birth_commit_fraction",
)


def _index(records: list[dict]) -> dict[str, dict]:
    indexed = {}
    for record in records:
        source_id = record.get("source_id")
        if not source_id or source_id in indexed:
            raise ValueError("reports require unique non-empty source IDs")
        indexed[source_id] = record
    if not indexed:
        raise ValueError("reports cannot be empty")
    return indexed


def _aggregate(records: list[dict], panel: str) -> dict:
    render = {}
    for metric in RENDER_METRICS:
        values = [
            record["render_signatures"][panel].get(metric) for record in records
        ]
        if all(value is not None for value in values):
            render[metric] = statistics.fmean(values)
    topology = {
        metric: statistics.fmean(
            record["topology_adherence"][panel][metric] for record in records
        )
        for metric in TOPOLOGY_METRICS
    }
    return {"render": render, "topology": topology}


def compare_realizer_reports(
    baseline: list[dict], candidate: list[dict], *, panel: str = "flow guided"
) -> dict:
    """Return paired per-source and macro deltas for one named render panel."""
    baseline_by_source = _index(baseline)
    candidate_by_source = _index(candidate)
    if set(baseline_by_source) != set(candidate_by_source):
        raise ValueError("baseline and candidate source sets do not match")
    source_ids = sorted(baseline_by_source)
    per_source = []
    for source_id in source_ids:
        before = baseline_by_source[source_id]
        after = candidate_by_source[source_id]
        if panel not in before.get("render_signatures", {}) or panel not in after.get(
            "render_signatures", {}
        ):
            raise ValueError(f"panel {panel!r} is absent for {source_id}")
        if panel not in before.get("topology_adherence", {}) or panel not in after.get(
            "topology_adherence", {}
        ):
            raise ValueError(f"topology panel {panel!r} is absent for {source_id}")
        before_render = before["render_signatures"][panel]
        after_render = after["render_signatures"][panel]
        shared_metrics = [
            metric
            for metric in RENDER_METRICS
            if before_render.get(metric) is not None
            and after_render.get(metric) is not None
        ]
        per_source.append(
            {
                "source_id": source_id,
                "class_name": after.get("class_name"),
                "baseline": {metric: before_render[metric] for metric in shared_metrics},
                "candidate": {metric: after_render[metric] for metric in shared_metrics},
                "delta": {
                    metric: after_render[metric] - before_render[metric]
                    for metric in shared_metrics
                },
            }
        )
    baseline_aggregate = _aggregate(
        [baseline_by_source[source_id] for source_id in source_ids], panel
    )
    candidate_aggregate = _aggregate(
        [candidate_by_source[source_id] for source_id in source_ids], panel
    )
    return {
        "panel": panel,
        "sources": source_ids,
        "baseline": baseline_aggregate,
        "candidate": candidate_aggregate,
        "delta": {
            "render": {
                metric: candidate_aggregate["render"][metric]
                - baseline_aggregate["render"][metric]
                for metric in RENDER_METRICS
                if metric in baseline_aggregate["render"]
                and metric in candidate_aggregate["render"]
            },
            "topology": {
                metric: candidate_aggregate["topology"][metric]
                - baseline_aggregate["topology"][metric]
                for metric in TOPOLOGY_METRICS
            },
        },
        "per_source": per_source,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--panel", default="flow guided")
    parser.add_argument("--out")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    baseline = json.loads(Path(args.baseline).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    report = compare_realizer_reports(baseline, candidate, panel=args.panel)
    rendered = json.dumps(report, indent=2)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
