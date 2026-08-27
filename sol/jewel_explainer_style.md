# `jewel_explainer_style.py`

## Purpose

Defines the original visual language for the technical Jewel explainer series: geometric canvases,
stable semantic colors, typography, smooth reveals, vector primitives, and evidence-image placement.
It evokes rigorous mathematical animation without copying another channel's branding.

## Components

### `CanvasTheme` / `DARK_THEME` / `LIGHT_THEME`

- **Does**: Freezes 1280×720 composition, local Avenir/SF Mono/STIX math typefaces, and complete
  high-contrast dark or eggshell palettes for reproducible renders.
- **Color contract**: Neutral colors and all semantic label colors are resolved through the active
  theme. Episode 2 therefore gets black copy and darker accessible label colors without changing
  the established dark appearance of the other five episodes.

### `clamp` / `smooth` / `reveal` / `lerp`

- **Does**: Supplies bounded interpolation and quintic motion with zero endpoint velocity and
  acceleration.

### `JewelCanvas`

- **Does**: Wraps Pillow drawing with series headers/footers, typography, glowing Jewel marks,
  arrows, partial paths, geometric shapes, and contained evidence images.
- **Theme rule**: `color` maps stable series constants into the active palette, while `blend`
  performs alpha presentation against that palette's own background rather than a global dark fill.
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
| Series identity | Semantic token/trajectory/evidence roles remain stable across themes | Palette remapping |
