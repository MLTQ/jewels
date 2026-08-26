"""Render source and token-only programs for the passing individual-Jewel language."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from sol.audit_jewel_casting_language import _featurizer, _render
from sol.factorized_jewel_casting_language import load_factorized_codebook
from sol.prompt_jewel_caster import active_tokens_to_features, encode_active_jewel_tokens
from sol.render_jewel_casting_language import _panel, _row
from sol.render_streaming_continuation import frame_points


def select_records(report: dict) -> list[dict]:
    """Select the lowest fitter seed for every protocol-owned validation source."""
    selected = {}
    expected = list(report["protocol"]["validation_sources"])
    for row in report["records"]:
        source = row["source_id"]
        if source in expected and (
            source not in selected or row["fit_seed"] < selected[source]["fit_seed"]
        ):
            selected[source] = row
    if set(selected) != set(expected):
        raise ValueError("report is missing a validation source")
    return [selected[source] for source in expected]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--width", type=int, default=150)
    args = parser.parse_args()
    if min(args.height, args.width) <= 0:
        raise ValueError("render dimensions must be positive")
    device = torch.device(args.device)
    report = json.loads(Path(args.report).read_text())
    records = select_records(report)
    codebook = load_factorized_codebook(args.codebook, device)
    state_to_features, _ = _featurizer()
    rows = []
    for row in records:
        checkpoint = torch.load(row["path"], map_location="cpu", weights_only=False)
        features = state_to_features(checkpoint["state"]).float().to(device)
        tokens = encode_active_jewel_tokens(features, codebook)
        token_features = active_tokens_to_features(features[:, :3], tokens, codebook)
        background = torch.as_tensor(checkpoint["info"]["background"]).float()
        total_frames = int(checkpoint["info"]["shape"][0])
        indices = torch.linspace(0, total_frames - 1, 3).round().long()
        points = frame_points(
            total_frames, indices, args.height, args.width, device=device
        )
        for label, candidate in (("continuous target", features), ("three-token Jewel", token_features)):
            rendered = _render(candidate, points, background).reshape(
                3, args.height, args.width, 3
            )
            panels = [
                _panel(frame, f"{row['style']} / {label} / t={int(index)}")
                for frame, index in zip(rendered, indices)
            ]
            rows.append(_row(panels))
        print("rendered individual language", row["source_id"], flush=True)
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
