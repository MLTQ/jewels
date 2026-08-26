"""Reusable animated diagram scenes for the Jewel explainer episodes."""

from __future__ import annotations

import math
from typing import Any, Callable

from PIL import Image, ImageDraw

from sol.jewel_explainer_episodes import Episode, Shot
from sol.jewel_explainer_style import (
    BACKGROUND,
    BACKGROUND_2,
    BLUE,
    FOREGROUND,
    GRID,
    MUTED,
    ORANGE,
    PINK,
    PURPLE,
    RED,
    TEAL,
    WIDTH,
    YELLOW,
    JewelCanvas,
    blend_hex,
    clamp,
    evenly_spaced,
    lerp,
    reveal,
)


PALETTE = (BLUE, TEAL, YELLOW, PINK, ORANGE, PURPLE, RED)

# Diagrams are composited only inside this band.  The header and footer are
# deliberately outside it so an animated shape can never cover explanatory copy.
CONTENT_TOP = 170
CONTENT_BOTTOM = 630


def _node(
    canvas: JewelCanvas,
    center: tuple[float, float],
    label: str,
    amount: float,
    *,
    color: str,
    width: float = 190,
    height: float = 82,
) -> None:
    if amount <= 0:
        return
    scale = 0.85 + 0.15 * amount
    half_w, half_h = width * scale / 2, height * scale / 2
    canvas.rounded_rect(
        (center[0] - half_w, center[1] - half_h, center[0] + half_w, center[1] + half_h),
        fill=BACKGROUND_2,
        outline=color,
        width=3,
        radius=16,
        alpha=amount,
    )
    canvas.wrapped_text(
        (center[0] - half_w + 14, center[1] - 13),
        label,
        max_width=round(width - 28),
        size=23,
        color=FOREGROUND,
        line_gap=2,
        alpha=amount,
    )


def _draw_pipeline(canvas: JewelCanvas, shot: Shot, progress: float, assets: dict[str, Any]) -> None:
    del assets
    nodes = list(shot.payload["nodes"])
    centers = list(evenly_spaced(len(nodes), 125, 1155))
    width = min(210, 810 / max(len(nodes) - 1, 1))
    for index, (label, x) in enumerate(zip(nodes, centers)):
        amount = reveal(progress, 0.05 + index * 0.11, 0.23 + index * 0.11)
        _node(canvas, (x, 345), label, amount, color=PALETTE[index], width=width)
        if index:
            arrow_amount = reveal(progress, 0.14 + (index - 1) * 0.11, 0.30 + (index - 1) * 0.11)
            canvas.arrow(
                (centers[index - 1] + width / 2 + 8, 345),
                (x - width / 2 - 8, 345),
                color=TEAL,
                width=4,
                alpha=arrow_amount,
            )
    if len(nodes) <= 4:
        canvas.text((640, 500), "causal direction", size=24, color=MUTED, anchor="ma", alpha=reveal(progress, 0.72, 0.92))


def _draw_exclusion(canvas: JewelCanvas, shot: Shot, progress: float, assets: dict[str, Any]) -> None:
    del assets
    allowed = shot.payload["allowed"]
    forbidden = shot.payload["forbidden"]
    canvas.text((280, 205), "ALLOWED", size=24, color=TEAL, family="mono", anchor="ma")
    canvas.text((920, 205), "FORBIDDEN", size=24, color=RED, family="mono", anchor="ma")
    for index, label in enumerate(allowed):
        amount = reveal(progress, 0.05 + index * 0.10, 0.23 + index * 0.10)
        y = 280 + index * 100
        _node(canvas, (280, y), label, amount, color=TEAL, width=300, height=70)
    for index, label in enumerate(forbidden):
        amount = reveal(progress, 0.16 + index * 0.08, 0.32 + index * 0.08)
        y = 250 + index * 76
        _node(canvas, (920, y), label, amount, color=RED, width=310, height=58)
        cross = reveal(progress, 0.29 + index * 0.08, 0.43 + index * 0.08)
        canvas.line(((800, y - 24), (1040, y + 24)), color=RED, width=5, alpha=cross)
    canvas.arrow((480, 380), (695, 380), color=YELLOW, width=5, alpha=reveal(progress, 0.70, 0.90))


def _draw_tokens(canvas: JewelCanvas, shot: Shot, progress: float, assets: dict[str, Any]) -> None:
    del assets
    tokens = shot.payload["tokens"]
    values = shot.payload["values"]
    centers = list(evenly_spaced(len(tokens), 245, 1035))
    for index, (token, value, x) in enumerate(zip(tokens, values, centers)):
        amount = reveal(progress, 0.07 + index * 0.13, 0.28 + index * 0.13)
        canvas.diamond((x, 315), 62 * amount, fill=PALETTE[index], outline=FOREGROUND, alpha=amount)
        canvas.text((x, 410), token, size=26, color=PALETTE[index], family="mono", anchor="ma", alpha=amount)
        canvas.text((x, 460), value, size=28, color=FOREGROUND, anchor="ma", alpha=amount)
        if index:
            canvas.arrow((centers[index - 1] + 76, 315), (x - 76, 315), color=MUTED, width=3, alpha=amount)


def _draw_program(canvas: JewelCanvas, shot: Shot, progress: float, assets: dict[str, Any]) -> None:
    del assets
    rows = shot.payload["rows"]
    canvas.rounded_rect((245, 185, 1035, 545), fill=BACKGROUND_2, outline=GRID, width=2)
    for index, (key, value) in enumerate(rows):
        amount = reveal(progress, 0.05 + index * 0.12, 0.25 + index * 0.12)
        y = 240 + index * 76
        color = PALETTE[index]
        canvas.text((300, y), key, size=25, color=color, family="mono", alpha=amount)
        canvas.text((580, y), value, size=29, color=FOREGROUND, alpha=amount)
        if index < len(rows) - 1:
            canvas.line(((295, y + 34), (985, y + 34)), color=GRID, width=1, alpha=amount)


def _cube(canvas: JewelCanvas, amount: float) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    front = [(310, 260), (760, 260), (760, 535), (310, 535), (310, 260)]
    offset = (135, -85)
    back = [(x + offset[0], y + offset[1]) for x, y in front]
    canvas.polyline_partial(front, amount, color=MUTED, width=3)
    canvas.polyline_partial(back, amount, color=BLUE, width=3)
    for index in range(4):
        canvas.line((front[index], back[index]), color=GRID, width=2, alpha=amount)
    return front, back


def _draw_spacetime(canvas: JewelCanvas, shot: Shot, progress: float, assets: dict[str, Any]) -> None:
    del assets
    mode = shot.payload.get("mode", "slice")
    cube_amount = reveal(progress, 0.02, 0.22)
    _cube(canvas, cube_amount)
    path = []
    for index in range(13):
        t = index / 12
        path.append((385 + 475 * t, 405 - 66 * math.sin(t * math.pi * 1.45)))
    path_amount = reveal(progress, 0.18, 0.52)
    canvas.polyline_partial(path, path_amount, color=YELLOW, width=7)
    for index in range(72):
        phase = ((index * 37) % 101) / 100
        t = ((index * 19) % 73) / 72
        px = 360 + 520 * t
        center_y = 405 - 66 * math.sin(t * math.pi * 1.45)
        spread = 38 + 92 * (((index * 11) % 17) / 16)
        py = center_y + spread * math.sin(index * 2.17)
        distance = abs(py - center_y)
        if mode in {"rank-balanced", "donors"}:
            foreground = distance < 65
            color = PINK if foreground else BLUE
        else:
            color = TEAL if index % 3 else BLUE
        amount = reveal(progress, 0.28 + phase * 0.32, 0.50 + phase * 0.32)
        canvas.glow_dot((px, py), 3.5, color=color, alpha=amount)
    if mode == "slice":
        x = lerp(385, 860, reveal(progress, 0.48, 0.92))
        canvas.draw.polygon(
            ((x, 190), (x + 135, 105), (x + 135, 380), (x, 465)),
            fill=blend_hex(BACKGROUND, TEAL, 0.18),
            outline=TEAL,
        )
        canvas.text((1045, 300), "frame t", size=30, color=TEAL, anchor="ma")
        canvas.arrow((1005, 325), (885, 360), color=TEAL, width=4)
    elif mode == "continuous":
        canvas.text((1015, 265), "routing cells", size=25, color=MUTED, anchor="ma")
        canvas.text((1015, 325), "continuous μ", size=30, color=TEAL, anchor="ma")
        canvas.text((1015, 380), "≠ cell centers", size=26, color=YELLOW, anchor="ma")
    else:
        canvas.text((1045, 270), "foreground", size=25, color=PINK, anchor="ma")
        canvas.text((1045, 325), "36,000", size=36, color=PINK, anchor="ma")
        canvas.text((1045, 415), "background", size=25, color=BLUE, anchor="ma")
        canvas.text((1045, 470), "36,000", size=36, color=BLUE, anchor="ma")
    canvas.text((280, 556), "u", size=24, color=MUTED)
    canvas.text((785, 557), "t", size=24, color=MUTED)
    canvas.text((885, 164), "v", size=24, color=MUTED)


def _draw_factorization(canvas: JewelCanvas, shot: Shot, progress: float, assets: dict[str, Any]) -> None:
    del assets
    factors = shot.payload["factors"]
    centers = list(evenly_spaced(len(factors), 190, 1090))
    for index, ((name, size_label, _), x) in enumerate(zip(factors, centers)):
        amount = reveal(progress, 0.05 + index * 0.12, 0.25 + index * 0.12)
        canvas.diamond((x, 300), 57 * amount, fill=PALETTE[index], outline=FOREGROUND, alpha=amount)
        canvas.text((x, 400), name, size=25, color=PALETTE[index], anchor="ma", alpha=amount)
        canvas.text((x, 448), size_label, size=24, color=FOREGROUND, family="mono", anchor="ma", alpha=amount)
        if index < len(factors) - 1:
            canvas.text(((x + centers[index + 1]) / 2, 300), "×", size=34, color=MUTED, family="math", anchor="mm", alpha=amount)
    canvas.text((640, 525), "aligned prototypes compose one renderable Jewel phrase", size=25, color=MUTED, anchor="ma", alpha=reveal(progress, 0.72, 0.94))


def _draw_feature_vector(canvas: JewelCanvas, shot: Shot, progress: float, assets: dict[str, Any]) -> None:
    del assets
    segments = shot.payload["segments"]
    total = sum(count for _, count, _ in segments)
    x0, x1 = 100, 1180
    x = x0
    for index, (label, count, _) in enumerate(segments):
        target_width = (x1 - x0) * count / total
        amount = reveal(progress, 0.04 + index * 0.12, 0.25 + index * 0.12)
        width = target_width * amount
        canvas.rounded_rect((x, 285, x + width, 390), fill=PALETTE[index], outline=PALETTE[index], radius=10, alpha=0.68 * amount)
        if width > 55:
            canvas.text((x + width / 2, 335), f"{label}\n{count}", size=22, color=FOREGROUND, family="mono", anchor="mm", alpha=amount)
        x += target_width
    canvas.text((100, 445), "index", size=22, color=MUTED, family="mono")
    canvas.text((1180, 445), str(total - 1), size=22, color=MUTED, family="mono", anchor="ra")
    canvas.arrow((150, 480), (1130, 480), color=GRID, width=3, alpha=reveal(progress, 0.70, 0.90))


def _draw_equation(canvas: JewelCanvas, shot: Shot, progress: float, assets: dict[str, Any]) -> None:
    del assets
    equation = shot.payload["equation"]
    amount = reveal(progress, 0.04, 0.30)
    size = 33 if len(equation) < 65 else 27
    canvas.rounded_rect((95, 185, 1185, 285), fill=BACKGROUND_2, outline=GRID, radius=16, alpha=amount)
    canvas.text((640, 236), equation, size=size, color=YELLOW, family="math", anchor="mm", alpha=amount)
    diagram = shot.payload.get("diagram")
    diagram_amount = reveal(progress, 0.30, 0.75)
    if diagram in {"ellipsoid", "distance", "gauge"}:
        canvas.ellipse((260, 345, 600, 500), outline=BLUE, width=5, alpha=diagram_amount)
        canvas.line(((430, 423), (650, 350)), color=PINK, width=5, alpha=diagram_amount)
        canvas.arrow((650, 350), (805, 300), color=PINK, width=4, alpha=diagram_amount)
        canvas.text((835, 295), "principal axis", size=25, color=PINK, alpha=diagram_amount)
        if diagram == "distance":
            canvas.glow_dot((430, 423), 7, color=YELLOW, alpha=diagram_amount)
            canvas.glow_dot((650, 350), 7, color=TEAL, alpha=diagram_amount)
            canvas.text((415, 535), "μ", size=28, color=YELLOW, family="math", anchor="ma")
            canvas.text((675, 380), "x", size=28, color=TEAL, family="math")
        if diagram == "gauge":
            canvas.text((930, 360), "q", size=36, color=TEAL, family="math", anchor="ma")
            canvas.text((930, 440), "−q", size=36, color=PINK, family="math", anchor="ma")
            canvas.text((930, 510), "same R", size=27, color=FOREGROUND, anchor="ma")
    elif diagram in {"gradient", "sum"}:
        for index, x in enumerate(evenly_spaced(7, 250, 750)):
            radius = 25 + index * 6
            canvas.circle((x, 420), radius, fill=PALETTE[index % 3], outline=PALETTE[index % 3], alpha=diagram_amount * 0.45)
        canvas.arrow((800, 420), (940, 420), color=FOREGROUND, width=4, alpha=diagram_amount)
        canvas.rounded_rect((955, 360, 1080, 480), fill=TEAL, outline=TEAL, alpha=diagram_amount * 0.5)
        canvas.text((1018, 420), "RGB", size=30, color=FOREGROUND, family="mono", anchor="mm", alpha=diagram_amount)
    elif diagram in {"path", "rank", "derivative"}:
        points = [(190 + index * 88, 470 - 95 * math.sin(index * 0.65)) for index in range(10)]
        canvas.polyline_partial(points, diagram_amount, color=YELLOW, width=7)
        for index, point in enumerate(points):
            if index % 2 == 0:
                canvas.glow_dot(point, 6, color=TEAL if diagram != "rank" else PINK, alpha=diagram_amount)
        if diagram == "derivative":
            canvas.arrow(points[4], (points[4][0] + 115, points[4][1] - 70), color=PINK, width=5, alpha=diagram_amount)
            canvas.text((points[4][0] + 125, points[4][1] - 75), "v[k]", size=28, color=PINK, family="math")
    canvas.text((640, 565), "every symbol maps directly to the implementation", size=23, color=MUTED, anchor="ma", alpha=reveal(progress, 0.72, 0.92))


def _draw_support(canvas: JewelCanvas, shot: Shot, progress: float, assets: dict[str, Any]) -> None:
    del assets
    amount = reveal(progress, 0.04, 0.28)
    for x in range(170, 900, 72):
        canvas.line(((x, 190), (x, 560)), color=GRID, width=1, alpha=amount)
    for y in range(200, 560, 72):
        canvas.line(((150, y), (900, y)), color=GRID, width=1, alpha=amount)
    canvas.ellipse((260, 320, 760, 430), outline=PINK, width=6, alpha=reveal(progress, 0.22, 0.52))
    canvas.rounded_rect((245, 295, 775, 455), fill=BACKGROUND, outline=YELLOW, width=3, radius=2, alpha=reveal(progress, 0.38, 0.65))
    point = (710, 375)
    canvas.glow_dot(point, 8, color=TEAL, alpha=reveal(progress, 0.52, 0.72))
    canvas.text((1010, 245), f"support = {shot.payload['sigma']}σ", size=30, color=YELLOW, anchor="ma")
    canvas.text((1010, 325), f"{shot.payload['neighbors']} neighbor cells", size=27, color=FOREGROUND, anchor="ma")
    canvas.text((1010, 405), f"boundary weight\n{shot.payload['boundary']}", size=27, color=MUTED, family="mono", anchor="ma")
    canvas.text((1010, 510), "true q test", size=27, color=TEAL, anchor="ma")


def _draw_comparison(canvas: JewelCanvas, shot: Shot, progress: float, assets: dict[str, Any]) -> None:
    del assets
    for index, side in enumerate(("left", "right")):
        x0 = 90 + index * 600
        color = RED if index == 0 else TEAL
        amount = reveal(progress, 0.05 + index * 0.18, 0.32 + index * 0.18)
        canvas.rounded_rect((x0, 205, x0 + 500, 520), fill=BACKGROUND_2, outline=color, width=3, alpha=amount)
        canvas.text((x0 + 250, 270), shot.payload[side], size=31, color=color, anchor="ma", alpha=amount)
        canvas.text((x0 + 250, 390), shot.payload[f"{side}_value"], size=47, color=FOREGROUND, family="mono", anchor="mm", alpha=amount)
        if index == 0:
            for mark in range(25):
                px = x0 + 90 + ((mark * 79) % 320)
                py = 320 + ((mark * 47) % 140)
                canvas.glow_dot((px, py), 3, color=PALETTE[mark % 5], alpha=0.45 * amount)
        else:
            path = [(x0 + 80 + k * 42, 430 - 80 * math.sin(k * 0.6)) for k in range(9)]
            canvas.polyline_partial(path, amount, color=YELLOW, width=5)


def _draw_evidence(canvas: JewelCanvas, shot: Shot, progress: float, assets: dict[str, Any]) -> None:
    image = assets.get(shot.payload["asset"])
    amount = reveal(progress, 0.05, 0.28)
    if isinstance(image, Image.Image):
        canvas.paste_contained(image, (80, 175, 1200, 545), alpha=amount)
    else:
        canvas.rounded_rect((80, 175, 1200, 545), fill=BACKGROUND_2, outline=RED)
        canvas.text((640, 360), "evidence asset unavailable", size=36, color=RED, anchor="mm")
    canvas.rounded_rect((330, 530, 950, 585), fill=BACKGROUND_2, outline=TEAL, radius=12, alpha=reveal(progress, 0.55, 0.80))
    canvas.text((640, 558), shot.payload["label"], size=23, color=TEAL, anchor="mm", alpha=reveal(progress, 0.55, 0.80))


def _draw_metrics(canvas: JewelCanvas, shot: Shot, progress: float, assets: dict[str, Any]) -> None:
    del assets
    bars = shot.payload["bars"]
    y_positions = list(evenly_spaced(len(bars), 230, 520))
    for index, ((label, value, maximum, color_index), y) in enumerate(zip(bars, y_positions)):
        amount = reveal(progress, 0.04 + index * 0.10, 0.28 + index * 0.10)
        color = PALETTE[color_index % len(PALETTE)]
        canvas.text((120, y), label, size=27, color=FOREGROUND, anchor="lm", alpha=amount)
        canvas.draw.rectangle((390, y - 18, 1050, y + 18), fill=GRID)
        width = 660 * clamp(value / maximum) * amount
        canvas.draw.rectangle((390, y - 18, 390 + width, y + 18), fill=color)
        if isinstance(value, int):
            formatted = str(value)
        elif value == 0:
            formatted = "0"
        elif value < 2:
            formatted = f"{value:.4f}"
        else:
            formatted = f"{value:.4f}"
        canvas.text((1080, y), formatted, size=25, color=color, family="mono", anchor="lm", alpha=amount)


def _draw_curve(canvas: JewelCanvas, shot: Shot, progress: float, assets: dict[str, Any]) -> None:
    del assets
    series = shot.payload["series"]
    left, top, right, bottom = 150, 210, 1010, 535
    canvas.line(((left, top), (left, bottom), (right, bottom)), color=MUTED, width=3, alpha=reveal(progress, 0.02, 0.20))
    canvas.text((580, 575), "evaluation step", size=23, color=MUTED, anchor="ma")
    canvas.text((76, 372), "token NLL", size=23, color=MUTED, anchor="mm")
    values = [value for _, rows, _ in series for value in rows]
    low, high = min(values) - 0.2, max(values) + 0.2
    for series_index, (name, rows, color_index) in enumerate(series):
        points = []
        for index, value in enumerate(rows):
            x = lerp(left + 30, right - 30, index / max(len(rows) - 1, 1))
            y = lerp(bottom - 25, top + 25, (value - low) / (high - low))
            points.append((x, y))
        amount = reveal(progress, 0.12 + series_index * 0.12, 0.55 + series_index * 0.12)
        canvas.polyline_partial(points, amount, color=PALETTE[color_index], width=5)
        if amount > 0.5:
            canvas.text((1030, points[-1][1]), name, size=24, color=PALETTE[color_index], anchor="lm", alpha=amount)
    best_amount = reveal(progress, 0.68, 0.90)
    canvas.line(((left + 30, top), (left + 30, bottom)), color=YELLOW, width=3, alpha=best_amount)
    canvas.text((left + 45, top + 8), "best = 100", size=23, color=YELLOW, alpha=best_amount)


def _draw_future(canvas: JewelCanvas, shot: Shot, progress: float, assets: dict[str, Any]) -> None:
    del assets
    stages = shot.payload["stages"]
    centers = list(evenly_spaced(len(stages), 115, 1165))
    width = min(205, 820 / max(len(stages) - 1, 1))
    for index, (stage, x) in enumerate(zip(stages, centers)):
        amount = reveal(progress, 0.04 + index * 0.10, 0.24 + index * 0.10)
        y = 315 + 45 * math.sin(index * math.pi)
        _node(canvas, (x, y), stage, amount, color=PALETTE[index], width=width, height=100)
        if index:
            canvas.arrow((centers[index - 1] + width / 2 + 5, y), (x - width / 2 - 5, y), color=YELLOW, width=4, alpha=amount)
    for index in range(40):
        phase = index / 39
        amount = reveal(progress, 0.58 + phase * 0.22, 0.76 + phase * 0.22)
        canvas.glow_dot((120 + phase * 1040, 500 + 24 * math.sin(index)), 3.5, color=TEAL, alpha=amount)


def _draw_playback(canvas: JewelCanvas, shot: Shot, progress: float, assets: dict[str, Any]) -> None:
    frames = assets.get(f"clip:{shot.payload['clip']}", [])
    if frames:
        index = min(len(frames) - 1, int(clamp(progress) * len(frames)))
        canvas.rounded_rect((120, 175, 810, 570), fill=BACKGROUND_2, outline=TEAL, width=3)
        canvas.paste_contained(frames[index], (145, 195, 785, 550), alpha=reveal(progress, 0.02, 0.18))
    for metric_index, metric in enumerate(shot.payload["metrics"]):
        amount = reveal(progress, 0.35 + metric_index * 0.13, 0.56 + metric_index * 0.13)
        canvas.rounded_rect((850, 230 + metric_index * 105, 1190, 305 + metric_index * 105), fill=BACKGROUND_2, outline=PALETTE[metric_index], width=2, alpha=amount)
        canvas.text((1020, 268 + metric_index * 105), metric, size=25, color=PALETTE[metric_index], anchor="mm", alpha=amount)


SCENE_RENDERERS: dict[str, Callable[[JewelCanvas, Shot, float, dict[str, Any]], None]] = {
    "pipeline": _draw_pipeline,
    "exclusion": _draw_exclusion,
    "tokens": _draw_tokens,
    "program": _draw_program,
    "spacetime": _draw_spacetime,
    "factorization": _draw_factorization,
    "feature-vector": _draw_feature_vector,
    "equation": _draw_equation,
    "support": _draw_support,
    "comparison": _draw_comparison,
    "evidence": _draw_evidence,
    "metrics": _draw_metrics,
    "curve": _draw_curve,
    "future": _draw_future,
    "playback": _draw_playback,
}


def draw_shot(
    episode: Episode,
    shot: Shot,
    shot_progress: float,
    episode_progress: float,
    assets: dict[str, Any],
) -> Image.Image:
    """Draw one complete video frame for a shot specification."""
    if shot.visual not in SCENE_RENDERERS:
        raise ValueError(f"unknown explainer visual {shot.visual!r}")
    canvas = JewelCanvas()
    canvas.header(episode.number, episode.title, shot.title)
    diagram = JewelCanvas()
    SCENE_RENDERERS[shot.visual](diagram, shot, clamp(shot_progress), assets)
    canvas.image.paste(
        diagram.image.crop((0, CONTENT_TOP, WIDTH, CONTENT_BOTTOM)),
        (0, CONTENT_TOP),
    )
    canvas.draw = ImageDraw.Draw(canvas.image)
    canvas.footer(shot.caption, episode_progress)
    return canvas.image
