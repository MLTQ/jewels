"""Render held-out targets and prompt-only generated Jewel programs over time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from sol.audit_jewel_casting_language import FieldRecord, _render, load_field_records
from sol.factorized_jewel_casting_language import load_factorized_codebook
from sol.prompt_jewel_caster import active_tokens_to_features
from sol.render_jewel_casting_language import _panel, _row
from sol.render_streaming_continuation import frame_points


ARMS = ("target", "correct", "shuffled", "null")


def select_target_records(
    records: list[FieldRecord], source_ids: list[str]
) -> list[FieldRecord]:
    """Select the lowest fitter seed for each report-owned validation source."""
    selected = []
    for source_id in source_ids:
        candidates = [row for row in records if row.source_id == source_id]
        if not candidates:
            raise ValueError(f"missing target field for {source_id}")
        selected.append(min(candidates, key=lambda row: row.fit_seed))
    return selected


def validate_programs(programs: dict, source_ids: list[str]) -> None:
    """Reject incomplete generated-program sheets before expensive rendering."""
    expected = {(source_id, arm) for source_id in source_ids for arm in ARMS[1:]}
    missing = expected - set(programs)
    if missing:
        raise ValueError(f"generated programs are incomplete: {sorted(missing)}")
    for key in expected:
        row = programs[key]
        if row["centers"].ndim != 2 or row["centers"].shape[1] != 3:
            raise ValueError(f"generated centers have invalid shape for {key}")
        if row["tokens"].shape != (len(row["centers"]), 3):
            raise ValueError(f"generated tokens have invalid shape for {key}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--programs", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--width", type=int, default=135)
    parser.add_argument("--frames", type=int, default=49)
    args = parser.parse_args()
    if min(args.height, args.width, args.frames) <= 0:
        raise ValueError("render dimensions and frame count must be positive")
    device = torch.device(args.device)
    report = json.loads(Path(args.report).read_text())
    source_ids = list(report["protocol"]["validation_sources"])
    records = load_field_records([Path(root) for root in args.root])
    targets = select_target_records(records, source_ids)
    programs = torch.load(args.programs, map_location="cpu", weights_only=False)
    validate_programs(programs, source_ids)
    codebook = load_factorized_codebook(args.codebook, device)
    indices = torch.linspace(0, args.frames - 1, 3).round().long()
    points = frame_points(
        args.frames, indices, args.height, args.width, device=device
    )
    rows = []
    for target in targets:
        candidates = {"target": target.features.to(device)}
        for arm in ARMS[1:]:
            program = programs[(target.source_id, arm)]
            candidates[arm] = active_tokens_to_features(
                program["centers"].to(device),
                program["tokens"].to(device),
                codebook,
            )
        prompt_label = target.source_id.split("__", 1)[0]
        for arm in ARMS:
            rendered = _render(
                candidates[arm], points, target.background
            ).reshape(3, args.height, args.width, 3)
            panels = [
                _panel(frame, f"{prompt_label} / {arm} / t={int(index)}")
                for frame, index in zip(rendered, indices)
            ]
            rows.append(_row(panels))
        print("rendered prompt", target.source_id, flush=True)
    sheet = Image.new(
        "RGB",
        (rows[0].width, sum(row.height for row in rows) + 3 * (len(rows) - 1)),
        "white",
    )
    offset = 0
    for row in rows:
        sheet.paste(row, (0, offset))
        offset += row.height + 3
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


if __name__ == "__main__":
    main()
