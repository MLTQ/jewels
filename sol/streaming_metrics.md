# `streaming_metrics.py`

## Purpose

Measures the streaming quantities hidden by a fixed jewel count per window and proves that
carry/commit rendering is equivalent to rendering the monolithic fitted field.

## Components

### `measure_streaming_contract`

- **Does**: reports active/effective density, initial state, continuation births, lifespan
  distribution, Little's-law identity, and per-window state sizes
- **Interacts with**: `streaming.py` and `splat_density.py`
- **Rationale**: the quality budget is active jewels per megapixel-frame; generator cost is births
  per megapixel-second; lifespan connects them

### `audit_carry_commit_render`

- **Does**: compares finite-support monolithic rendering against rendering each committed stride
  from its carried-plus-born active subset
- **Interacts with**: `render_truncated` in `render.py`
- **Rationale**: deterministic equivalence is required before training a continuation prior

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `audit_streaming_contract.py` | Returned report is JSON-safe | Report schema |
| Streaming experiment | Every sampled point is committed exactly once | Window coverage semantics |
| Future continuation prior | Density, birth rate, and lifespan remain separately reported | Collapsing metrics into total count |

## Notes

- “Birth” is the first observed finite-support frame. Jewels already active at frame zero are
  reported separately as initial state rather than continuation emissions.
- Megapixel normalization makes density comparable across resolutions; it does not demand spatially
  uniform allocation.
