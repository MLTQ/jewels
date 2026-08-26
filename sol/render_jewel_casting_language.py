"""Render source and decoded Jewel casting programs across spacetime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from sol.audit_jewel_casting_language import _featurizer, _render
from sol.jewel_casting_language import (
    CastingNormalizer,
    MotifCodebook,
    decode_program,
    encode_program,
    quantize_centers_to_cells,
)
from sol.render_streaming_continuation import frame_points
from sol.token_grid import GridSpec


def select_qualitative_records(report: dict) -> list[dict]:
    """Choose the lowest fit seed for every registered validation source."""
    sizes = sorted(int(key) for key in report["vocabularies"])
    if not sizes:
        raise ValueError("casting-language report has no vocabularies")
    expected = set(report["protocol"]["validation_sources"])
    selected: dict[str, dict] = {}
    for row in report["vocabularies"][str(sizes[-1])]["records"]:
        source = row["source_id"]
        if source not in expected:
            continue
        if source not in selected or row["fit_seed"] < selected[source]["fit_seed"]:
            selected[source] = row
    if set(selected) != expected:
        raise ValueError("report does not contain every validation source")
    return [selected[source] for source in sorted(selected)]


def _load_codebook(path: Path, device: torch.device) -> MotifCodebook:
    saved = torch.load(path, map_location=device, weights_only=False)
    return MotifCodebook(
        prototypes=saved["prototypes"].to(device),
        prototype_count_coordinates=saved["prototype_count_coordinates"].to(device),
        normalizer=CastingNormalizer(
            saved["normalizer"]["intrinsic_mean"].to(device),
            saved["normalizer"]["intrinsic_std"].to(device),
        ),
        grid_shape=tuple(saved["grid_shape"]),
        bundle_size=int(saved["bundle_size"]),
        count_weight=float(saved["count_weight"]),
    )


def _panel(frame: torch.Tensor, label: str) -> Image.Image:
    pixels = (frame.clamp(0, 1) * 255).round().byte().cpu().numpy()
    image = Image.fromarray(pixels)
    canvas = Image.new("RGB", (image.width, image.height + 24), "black")
    canvas.paste(image, (0, 24))
    ImageDraw.Draw(canvas).text((5, 6), label, fill="white")
    return canvas


def _row(panels: list[Image.Image], pad: int = 3) -> Image.Image:
    width = sum(panel.width for panel in panels) + pad * (len(panels) - 1)
    row = Image.new("RGB", (width, panels[0].height), "white")
    offset = 0
    for panel in panels:
        row.paste(panel, (offset, 0))
        offset += panel.width + pad
    return row


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--width", type=int, default=180)
    args = parser.parse_args()
    if min(args.height, args.width) <= 0:
        raise ValueError("render dimensions must be positive")
    device = torch.device(args.device)
    report_path = Path(args.report)
    report = json.loads(report_path.read_text())
    records = select_qualitative_records(report)
    vocabulary_size = max(int(key) for key in report["vocabularies"])
    codebook = _load_codebook(
        report_path.parent / f"codebook_k{vocabulary_size}.pt", device
    )
    spec = GridSpec(codebook.grid_shape, slots_per_cell=1)
    state_to_features, _ = _featurizer()
    rows = []
    for record in records:
        checkpoint = torch.load(
            record["path"], map_location="cpu", weights_only=False
        )
        features = state_to_features(checkpoint["state"]).float().to(device)
        background = torch.as_tensor(checkpoint["info"]["background"]).float()
        total_frames = int(checkpoint["info"]["shape"][0])
        indices = torch.linspace(0, total_frames - 1, 3).round().long()
        points = frame_points(
            total_frames, indices, args.height, args.width, device=device
        )
        program = encode_program(features, codebook)
        candidates = {
            "continuous source": features,
            "motif only": decode_program(program, codebook, residual_scale=0.0),
            "motif + 50% residual": decode_program(
                program, codebook, residual_scale=0.5
            ),
            "full residual": decode_program(program, codebook, residual_scale=1.0),
            "grid-center control": quantize_centers_to_cells(features, spec),
        }
        style = record.get("style", "unknown")
        for condition, candidate in candidates.items():
            rendered = _render(candidate, points, background).reshape(
                3, args.height, args.width, 3
            )
            panels = [
                _panel(frame, f"{style} / {condition} / t={int(index)}")
                for frame, index in zip(rendered, indices)
            ]
            rows.append(_row(panels))
        print("rendered", record["source_id"], flush=True)

    sheet = Image.new(
        "RGB", (rows[0].width, sum(row.height for row in rows) + 3 * (len(rows) - 1)),
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
