# `scaffold_appearance_adapter_rollout.py`

## Purpose

Runs a paired autonomous rollout in which the selected frozen mark flow owns topology, lifecycle,
geometry, density, carry selection, and stable IDs while a compact adapter can change only RGB.

## Components

### `AppearanceAdapterWindowReport`

- **Does**: Audits non-RGB equality in standardized, projected-local, and serialized-global units,
  plus the RGB residual and realized scaffold-gate coverage for every stride.

### `AppearanceAdapterRollout`

- **Does**: Couples two ordinary append-only rollout payloads and exposes exact topology, ID,
  lifecycle, and non-appearance ownership checks.

### `rollout_scaffold_appearance_adapter`

- **Does**: Predicts topology and causal row selections from the frozen field, constructs separate
  base/adapted RGB context rasters over those same rows, samples shared-noise mark pairs, and appends
  them with one ID sequence.
- **Rationale**: Adapted colors may support color continuity in the next stride, but they can never
  feed topology, activity, or row ownership.
- **Coordinate contract**: Non-RGB features are copied from the base after standardized sampling,
  local topology projection, and local-to-global conversion.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Deterministic render gate | Frozen arm matches ordinary base rollout for the same seed | RNG/order |
| Interactive editor | Both fields share append-only contiguous IDs | Row ownership |
| Density audit | Covariance, opacity, and lifecycle are bit-identical | Mutable dimensions |
| Continuation | Adapted context differs only through RGB statistics | Context policy |
| Saliency gate | One cell-weight raster per scaffold stride | Gate alignment |

## Notes

- This is intentionally narrower than `lifecycle_appearance_rollout.py`, whose screened second flow
  could change several spatial/appearance dimensions.  Here only canonical RGB 9--11 are mutable.
