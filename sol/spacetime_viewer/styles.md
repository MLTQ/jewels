# `styles.css`

## Purpose

Provides the viewer's responsive eggshell presentation, black text, compact transport, and
unobtrusive overlays. The visual language matches the revised Jewel explainer rather than the old
dark technical theme.

## Components

### `.viewer-shell` and `.stage-wrap`

- **Does**: Give the 3D field the majority of the viewport while keeping controls visible.
- **Rationale**: The stage is capped at 56% of viewport height so the play button remains visible on
  a typical laptop display.
- **Interacts with**: `index.html` and the canvas resize observer in `main.js`.

### `.transport`

- **Does**: Reflows the play button, slider, frame count, and one-line explanation on narrow screens.

### `.legend` and `.axis-label`

- **Does**: Explain the centroid, covariance-shell, time-axis, and playhead encodings without
  obscuring the volume.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `index.html` | Existing class names and responsive layout | Renaming selectors |
| Visual QA | Minimum 400px stage, above-fold desktop transport, and native focus outlines | Removing responsive rules |
