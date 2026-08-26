"""Render target and newly generated exact-prompt source videos side by side."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from sol.render_jewel_casting_language import _panel, _row
from stprim.data.video_io import load_video


def match_sources(target_manifest: dict, exact_manifest: dict) -> list[tuple[dict, list[dict]]]:
    """Match repeated exact-text sources to each held-out target by style and prompt."""
    exact = exact_manifest["examples"]
    keys = sorted({(row["style"], row["source_prompt"]) for row in exact})
    matches = []
    for key in keys:
        new_rows = [
            row for row in exact
            if (row["style"], row["source_prompt"]) == key
        ]
        targets = [
            row for row in target_manifest["examples"]
            if row.get("split") == "train"
            and (row["style"], row["source_prompt"]) == key
        ]
        if len(new_rows) < 2 or len(targets) != 1:
            raise ValueError(
                f"exact-prompt source match requires one target and at least two new videos: {key}"
            )
        matches.append((targets[0], sorted(new_rows, key=lambda row: row["source_id"])))
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--exact-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=270)
    args = parser.parse_args()
    if min(args.height, args.width) <= 0:
        raise ValueError("render dimensions must be positive")
    target_manifest = json.loads(Path(args.target_manifest).read_text())
    exact_manifest = json.loads(Path(args.exact_manifest).read_text())
    matches = match_sources(target_manifest, exact_manifest)
    rows = []
    for target, new_sources in matches:
        panels = []
        for label, item in [("held-out target", target)] + [
            (f"new exact-prompt source {index + 1}", row)
            for index, row in enumerate(new_sources)
        ]:
            video = load_video(
                item["video"], max_frames=49, start_frame=0,
                resize=(args.height, args.width), device="cpu",
            )
            panels.append(_panel(video[len(video) // 2], f"{item['style']} / {label}"))
        rows.append(_row(panels))
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
