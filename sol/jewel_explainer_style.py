"""Original vector-drawing language for the narrated Jewel explainer series."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1280
HEIGHT = 720

BACKGROUND = "#0b1020"
BACKGROUND_2 = "#111a30"
FOREGROUND = "#f3f0df"
MUTED = "#9aa6bb"
GRID = "#25314a"
BLUE = "#58a6ff"
TEAL = "#5ce1c4"
YELLOW = "#ffd166"
PINK = "#ff70a6"
ORANGE = "#ff9f43"
RED = "#ff6b6b"
PURPLE = "#b794f4"

BODY_FONT = Path("/System/Library/Fonts/Avenir.ttc")
MONO_FONT = Path("/System/Library/Fonts/SFNSMono.ttf")
MATH_FONT = Path("/System/Library/Fonts/Supplemental/STIXTwoMath.otf")

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def smooth(value: float) -> float:
    """Quintic ease with zero first and second derivatives at each endpoint."""
    x = clamp(value)
    return x * x * x * (x * (x * 6 - 15) + 10)


def reveal(progress: float, start: float, end: float) -> float:
    """Map global shot progress into one smooth local reveal."""
    if end <= start:
        raise ValueError("reveal interval must have positive width")
    return smooth((progress - start) / (end - start))


def lerp(first: float, second: float, amount: float) -> float:
    return first + (second - first) * clamp(amount)


def blend_hex(first: str, second: str, amount: float) -> str:
    amount = clamp(amount)
    a = tuple(int(first[index : index + 2], 16) for index in (1, 3, 5))
    b = tuple(int(second[index : index + 2], 16) for index in (1, 3, 5))
    values = tuple(round(lerp(x, y, amount)) for x, y in zip(a, b))
    return "#" + "".join(f"{value:02x}" for value in values)


class JewelCanvas:
    """Pillow-backed antialiased diagram canvas with a stable series palette."""

    def __init__(self, width: int = WIDTH, height: int = HEIGHT) -> None:
        self.width = width
        self.height = height
        self.image = Image.new("RGB", (width, height), BACKGROUND)
        self.draw = ImageDraw.Draw(self.image)
        self._background_grid()

    def _background_grid(self) -> None:
        for x in range(-self.height, self.width + self.height, 64):
            self.draw.line((x, 0, x - self.height, self.height), fill=GRID, width=1)
        self.draw.rectangle((0, 0, self.width, 86), fill=BACKGROUND_2)

    def font(self, size: int, *, family: str = "body") -> ImageFont.FreeTypeFont:
        key = (family, size)
        if key not in _FONT_CACHE:
            path = {"body": BODY_FONT, "mono": MONO_FONT, "math": MATH_FONT}[family]
            _FONT_CACHE[key] = ImageFont.truetype(str(path), size=size)
        return _FONT_CACHE[key]

    def text(
        self,
        xy: tuple[float, float],
        value: str,
        *,
        size: int = 32,
        color: str = FOREGROUND,
        family: str = "body",
        anchor: str = "la",
        alpha: float = 1.0,
    ) -> None:
        self.draw.text(
            xy,
            value,
            font=self.font(size, family=family),
            fill=blend_hex(BACKGROUND, color, alpha),
            anchor=anchor,
        )

    def text_width(self, value: str, *, size: int, family: str = "body") -> float:
        return self.draw.textlength(value, font=self.font(size, family=family))

    def wrapped_text(
        self,
        xy: tuple[float, float],
        value: str,
        *,
        max_width: int,
        size: int = 28,
        color: str = FOREGROUND,
        family: str = "body",
        line_gap: int = 8,
        alpha: float = 1.0,
    ) -> int:
        words = value.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and self.text_width(candidate, size=size, family=family) > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        y = xy[1]
        for line in lines:
            self.text((xy[0], y), line, size=size, color=color, family=family, alpha=alpha)
            y += size + line_gap
        return len(lines)

    def line(
        self,
        points: Sequence[tuple[float, float]],
        *,
        color: str = FOREGROUND,
        width: int = 3,
        alpha: float = 1.0,
    ) -> None:
        self.draw.line(points, fill=blend_hex(BACKGROUND, color, alpha), width=width, joint="curve")

    def arrow(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str = FOREGROUND,
        width: int = 4,
        alpha: float = 1.0,
    ) -> None:
        self.line((start, end), color=color, width=width, alpha=alpha)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        length = 13 + width
        wing = 0.52
        head = [
            end,
            (end[0] - length * math.cos(angle - wing), end[1] - length * math.sin(angle - wing)),
            (end[0] - length * math.cos(angle + wing), end[1] - length * math.sin(angle + wing)),
        ]
        self.draw.polygon(head, fill=blend_hex(BACKGROUND, color, alpha))

    def rounded_rect(
        self,
        box: tuple[float, float, float, float],
        *,
        fill: str = BACKGROUND_2,
        outline: str = GRID,
        width: int = 2,
        radius: int = 18,
        alpha: float = 1.0,
    ) -> None:
        self.draw.rounded_rectangle(
            box,
            radius=radius,
            fill=blend_hex(BACKGROUND, fill, alpha),
            outline=blend_hex(BACKGROUND, outline, alpha),
            width=width,
        )

    def circle(
        self,
        center: tuple[float, float],
        radius: float,
        *,
        fill: str | None = None,
        outline: str = FOREGROUND,
        width: int = 3,
        alpha: float = 1.0,
    ) -> None:
        box = (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        )
        self.draw.ellipse(
            box,
            fill=None if fill is None else blend_hex(BACKGROUND, fill, alpha),
            outline=blend_hex(BACKGROUND, outline, alpha),
            width=width,
        )

    def ellipse(
        self,
        box: tuple[float, float, float, float],
        *,
        fill: str | None = None,
        outline: str = FOREGROUND,
        width: int = 3,
        alpha: float = 1.0,
    ) -> None:
        self.draw.ellipse(
            box,
            fill=None if fill is None else blend_hex(BACKGROUND, fill, alpha),
            outline=blend_hex(BACKGROUND, outline, alpha),
            width=width,
        )

    def diamond(
        self,
        center: tuple[float, float],
        radius: float,
        *,
        fill: str = BLUE,
        outline: str = FOREGROUND,
        alpha: float = 1.0,
    ) -> None:
        x, y = center
        self.draw.polygon(
            ((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)),
            fill=blend_hex(BACKGROUND, fill, alpha * 0.62),
            outline=blend_hex(BACKGROUND, outline, alpha),
        )

    def glow_dot(
        self,
        center: tuple[float, float],
        radius: float,
        *,
        color: str,
        alpha: float = 1.0,
    ) -> None:
        x, y = center
        self.draw.ellipse(
            (x - radius * 3, y - radius * 3, x + radius * 3, y + radius * 3),
            fill=blend_hex(BACKGROUND, color, alpha * 0.12),
        )
        self.draw.ellipse(
            (x - radius * 1.8, y - radius * 1.8, x + radius * 1.8, y + radius * 1.8),
            fill=blend_hex(BACKGROUND, color, alpha * 0.28),
        )
        self.draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=blend_hex(BACKGROUND, color, alpha),
        )

    def polyline_partial(
        self,
        points: Sequence[tuple[float, float]],
        amount: float,
        *,
        color: str,
        width: int = 4,
    ) -> None:
        if len(points) < 2 or amount <= 0:
            return
        amount = clamp(amount)
        segment_lengths = [
            math.dist(points[index], points[index + 1]) for index in range(len(points) - 1)
        ]
        target = sum(segment_lengths) * amount
        output = [points[0]]
        consumed = 0.0
        for index, length in enumerate(segment_lengths):
            if consumed + length <= target:
                output.append(points[index + 1])
                consumed += length
                continue
            fraction = (target - consumed) / max(length, 1e-8)
            output.append((
                lerp(points[index][0], points[index + 1][0], fraction),
                lerp(points[index][1], points[index + 1][1], fraction),
            ))
            break
        if len(output) >= 2:
            self.line(output, color=color, width=width)

    def paste_contained(
        self,
        image: Image.Image,
        box: tuple[int, int, int, int],
        *,
        alpha: float = 1.0,
    ) -> None:
        x0, y0, x1, y1 = box
        candidate = image.convert("RGB")
        scale = min((x1 - x0) / candidate.width, (y1 - y0) / candidate.height)
        candidate = candidate.resize(
            (max(1, round(candidate.width * scale)), max(1, round(candidate.height * scale))),
            Image.Resampling.LANCZOS,
        )
        x = x0 + (x1 - x0 - candidate.width) // 2
        y = y0 + (y1 - y0 - candidate.height) // 2
        if alpha < 1:
            candidate = Image.blend(Image.new("RGB", candidate.size, BACKGROUND), candidate, alpha)
        self.image.paste(candidate, (x, y))
        self.draw = ImageDraw.Draw(self.image)

    def header(self, episode_number: int, episode_title: str, shot_title: str) -> None:
        self.text((38, 42), f"JEWEL FIELD · {episode_number:02d}", size=19, color=TEAL, family="mono")
        self.text((330, 43), episode_title, size=22, color=MUTED)
        self.text((38, 116), shot_title, size=46, color=FOREGROUND)

    def footer(self, caption: str, progress: float) -> None:
        self.draw.rectangle((0, 630, self.width, self.height), fill=BACKGROUND_2)
        self.wrapped_text(
            (42, 650), caption, max_width=1140, size=23, color=FOREGROUND, line_gap=4
        )
        self.draw.rectangle((0, self.height - 5, self.width, self.height), fill=GRID)
        self.draw.rectangle((0, self.height - 5, round(self.width * clamp(progress)), self.height), fill=TEAL)


def evenly_spaced(count: int, start: float, end: float) -> Iterable[float]:
    if count <= 0:
        return ()
    if count == 1:
        return ((start + end) / 2,)
    return (lerp(start, end, index / (count - 1)) for index in range(count))
