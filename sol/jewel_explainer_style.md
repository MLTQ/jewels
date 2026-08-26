# `jewel_explainer_style.py`

## Purpose

Defines the original visual language for the technical Jewel explainer series: dark geometric
canvas, stable semantic colors, typography, smooth reveals, vector primitives, and evidence-image
placement. It evokes rigorous mathematical animation without copying another channel's branding.

## Components

### Palette and font constants

- **Does**: Freezes 1280×720 composition, high-contrast colors, and local Avenir/SF Mono/STIX math
  typefaces for reproducible renders.

### `clamp` / `smooth` / `reveal` / `lerp`

- **Does**: Supplies bounded interpolation and quintic motion with zero endpoint velocity and
  acceleration.

### `JewelCanvas`

- **Does**: Wraps Pillow drawing with series headers/footers, typography, glowing Jewel marks,
  arrows, partial paths, geometric shapes, and contained evidence images.
- **Image rule**: Contained images scale up or down to the largest aspect-preserving size inside
  their panel. This keeps low-resolution proof clips legible without stretching or cropping them.
- **Rationale**: A small native vector layer avoids adding Manim/Cairo dependencies to the project.
- **Performance contract**: Reuses a process-wide font cache and approximates point glow with
  concentric color blends so long episodes do not allocate and blur a full-frame layer per Jewel.

### `evenly_spaced`

- **Does**: Produces stable layout coordinates for repeated tokens and marks.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `jewel_explainer_scenes.py` | Palette, easing, and `JewelCanvas` drawing methods | Names/signatures |
| Renderer tests | Easing remains bounded and frame size remains 1280×720 | Math/canvas constants |
| Series identity | Token/trajectory/evidence colors remain stable across episodes | Palette remapping |
