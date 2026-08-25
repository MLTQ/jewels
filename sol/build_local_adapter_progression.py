"""Build pitch-readable strips from a labeled convergence audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _centered(
    draw: ImageDraw.ImageDraw,
    box_left: int,
    box_width: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str = "white",
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    draw.text((box_left + (box_width - width) / 2, y), text, font=font, fill=fill)


def build_progression(audit: Path, out: Path, style_dir: Path) -> dict:
    """Remove the lattice column and add exact metrics to a progression sheet."""
    report = json.loads((audit / "report.json").read_text())
    labels = report["protocol"].get("candidate_labels")
    if not labels:
        raise ValueError("progression audit needs candidate labels")
    styles = sorted({row["style"] for row in report["perceptual_records"]})
    image = Image.open(audit / "qualitative.png").convert("RGB")
    input_columns = len(labels) + 3  # target, lattice, candidates, teacher
    if image.width % input_columns or image.height % len(styles):
        raise ValueError("qualitative sheet does not match report dimensions")
    tile_width = image.width // input_columns
    tile_height = image.height // len(styles)
    keep_columns = [0, *range(2, 2 + len(labels)), input_columns - 1]
    output_width = tile_width * len(keep_columns)

    style_dir.mkdir(parents=True, exist_ok=True)
    strips = []
    for row, style in enumerate(styles):
        strip = Image.new("RGB", (output_width, tile_height), "black")
        for output_column, input_column in enumerate(keep_columns):
            tile = image.crop((
                input_column * tile_width,
                row * tile_height,
                (input_column + 1) * tile_width,
                (row + 1) * tile_height,
            ))
            strip.paste(tile, (output_column * tile_width, 0))
        strip.save(style_dir / f"{style}.png")
        strips.append(strip)

    header_height = 68
    footer_height = 34
    pitch = Image.new(
        "RGB",
        (output_width, header_height + tile_height * len(strips) + footer_height),
        "#11151c",
    )
    draw = ImageDraw.Draw(pitch)
    title_font = _font(15)
    metric_font = _font(12)
    _centered(draw, 0, tile_width, 14, "HELD-OUT TARGET", title_font)
    for index, label in enumerate(labels):
        left = (index + 1) * tile_width
        row = report["perceptual_macro"][f"irregular_seed{index}"]
        _centered(draw, left, tile_width, 8, label.upper(), title_font)
        _centered(
            draw,
            left,
            tile_width,
            35,
            f"LPIPS {row['lpips']:.4f}  |  PSNR {row['psnr']:.2f}",
            metric_font,
            fill="#d8dee9",
        )
    _centered(
        draw,
        (len(labels) + 1) * tile_width,
        tile_width,
        14,
        "FITTED CEILING",
        title_font,
    )
    for row, strip in enumerate(strips):
        pitch.paste(strip, (0, header_height + row * tile_height))
    footer_y = header_height + tile_height * len(strips) + 8
    _centered(
        draw,
        0,
        output_width,
        footer_y,
        "Same held-out clips • same frame • same renderer • only optimizer updates change",
        metric_font,
        fill="#d8dee9",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pitch.save(out)
    return {
        "styles": styles,
        "candidate_labels": labels,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "output_width": output_width,
        "output_height": pitch.height,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--style-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    evidence = build_progression(Path(args.audit), out, Path(args.style_dir))
    out.with_suffix(".json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
