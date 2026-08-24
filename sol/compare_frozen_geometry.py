"""Prove candidate geometry state tensors are bitwise equal to one source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def geometry_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Select the factorized encoder tensors that own predicted geometry."""
    selected = {
        name: value
        for name, value in state.items()
        if name.startswith(("geometry_trunk.", "geometry_head."))
    }
    if not selected:
        raise ValueError("checkpoint contains no factorized geometry tensors")
    return selected


def compare_geometry_states(
    source: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
) -> dict[str, object]:
    """Return exact equality, maximum change, and any mismatched tensor names."""
    source_geometry = geometry_state(source)
    candidate_geometry = geometry_state(candidate)
    if source_geometry.keys() != candidate_geometry.keys():
        raise ValueError("candidate geometry tensor names do not match source")
    mismatched = []
    maximum = 0.0
    for name, source_value in source_geometry.items():
        candidate_value = candidate_geometry[name]
        if source_value.shape != candidate_value.shape:
            raise ValueError(f"candidate geometry shape changed for {name}")
        if not torch.equal(source_value, candidate_value):
            mismatched.append(name)
        maximum = max(
            maximum,
            float((source_value - candidate_value).abs().max()),
        )
    return {
        "bitwise_exact": not mismatched,
        "max_abs_change": maximum,
        "tensor_count": len(source_geometry),
        "mismatched_tensors": mismatched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source_path = Path(args.source)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    records = []
    for path_text in args.candidate:
        path = Path(path_text)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        records.append({
            "checkpoint": str(path.resolve()),
            **compare_geometry_states(source["model"], saved["model"]),
        })
    report = {
        "schema": "frozen-geometry-checkpoint-comparison-v1",
        "source": str(source_path.resolve()),
        "all_bitwise_exact": all(row["bitwise_exact"] for row in records),
        "candidates": records,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
