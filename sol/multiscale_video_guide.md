# `multiscale_video_guide.py`

## Purpose

Preserves within-cell appearance and motion from a low-resolution video scaffold instead of
collapsing every jewel cell to one mean RGB value.

## Components

### `video_to_multiscale_cell_tokens`

- **Does**: Builds fine and progressively low-pass video volumes, samples a fixed local subgrid in
  every `(u,v,t)` birth cell, and emits RGB, RGB derivatives, local offsets, and scale identity.
- **Interacts with**: `GridSpec` for canonical cell order and `BirthMarkFlowModel` for local
  cell/rank cross-attention.
- **Rationale**: The v1 oracle guide improved macro geometry but gave all ranks in one cell the same
  three RGB values. Multiple positioned tokens let different jewels bind to different edges,
  colors, and motion inside that cell while coarse scales retain object context.

### `_difference` / `_cell_tokens`

- **Does**: Computes non-wrapping temporal/spatial RGB derivatives and converts dense `(t,v,u)`
  volumes into the canonical flattened `(u,v,t)` cell convention.
- **Rationale**: Explicit derivatives expose small motion and boundaries without requiring the
  per-jewel attention layer to infer neighbor differences from a pooled value.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Mark-flow model | Feature dimension is 16 | Channel additions or reordering |
| Guide tests | Cells flatten `(u,v,t)` and local tokens `(scale,u,v,t)` | Axis convention |
| Checkpoints | Scales and subgrid are stored in trainer arguments | Sampling geometry |

## Notes

- Default `(1,2,4)` scales and a `2×2×2` subgrid produce 24 tokens per birth cell.
- Scale 1 preserves the available scaffold sampling density; larger scales are explicit low-pass
  context, not extra source detail.
