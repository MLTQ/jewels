# `streaming_data.py`

## Purpose

Turns one continuously fitted field into reusable prefix/future training pairs. Carried jewels stay
in global coordinates and bit-identical; only new births become learned targets.

## Components

### `FeatureStandardizer`

- **Does**: normalizes variable-size context or birth sets and restores physical features

### `BirthTarget`

- **Does**: stores canonical sparse `(cell,rank)` birth marks, counts, stable IDs, and birth frames
- **Rationale**: cells are indexed by spatial center and first active time, not temporal center

### `birth_cells` / `pack_births`

- **Does**: exposes the canonical first-active-frame cell assignment and stable within-cell rank
  packing for both continuation and scaffold-topology datasets
- **Rationale**: topology generation must emit exactly the same cell/rank contract consumed by the
  frozen mark realizer; duplicating the private packing logic would risk a silent ordering fork

### `ContinuationView` / `ContinuationDataset`

- **Does**: couples a clamped prefix, exact carried state, sparse birth target, and complete future
  active set with the shared grid/time contracts

### `rasterize_context`

- **Does**: pools normalized prefix jewels into per-cell mean, variance, log-count, and occupancy
  channels for a bounded-cost convolutional context encoder
- **Rationale**: repeatedly applying a learned MLP to 50k prefix jewels would dominate the overfit

### `build_continuation_dataset`

- **Does**: derives complete 32-prefix/16-future pairs from monolithic features and fits separate
  context/birth normalization
- **Interacts with**: `streaming.py`, `streaming_features.py`, and `streaming_model.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Continuation trainer | `carried_ids + birth_ids` partition `active_commit_ids` | Ownership policy |
| Context encoder | Raster channels are `[mean(22), variance(22), log_count, occupied]` | Channel order |
| Birth decoder | Canonical rank is stable within `(u,v,birth-time)` cells | Sort or cell semantics |
| Scaffold topology model | Public packing uses the continuation model's exact cell/rank order | Function semantics |
| Renderer evaluation | Carried features remain global and birth values are frontier-local | Coordinate convention |

## Notes

- A jewel may have a temporal center beyond the current commit while its finite support begins in
  the commit. Birth-time bucketing therefore must not constrain the predicted temporal center to the
  cell interval.
