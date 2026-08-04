# `test_render_prior_samples.py`

## Purpose

Protects coordinate ordering for visual prior artifacts so selected-frame rendering remains aligned
with `(T,H,W,3)` video layout.

## Components

### `PriorRenderTests`
- **Does**: Checks normalized corners and selected temporal coordinates from `frame_points`.
- **Interacts with**: `render_prior_samples.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Comparison GIFs | Render points map frame, row, and column to `(t,v,u)` consistently | Grid order |
