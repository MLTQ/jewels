# `streaming_features.py`

## Purpose

Converts canonical jewel parameters between one monolithic fitted timeline and a continuation-local
time coordinate measured in future-stride units. Spatial coordinates remain unchanged.

## Components

### `to_frontier_time`

- **Does**: maps the continuation frontier to local time zero and one future stride to unit length
- **Interacts with**: `streaming_data.py` for reusable prefix/future targets
- **Rationale**: absolute 96-frame normalization must not leak the window index into the model

### `to_global_time`

- **Does**: returns predicted local jewels to the monolithic fitted coordinate system for rendering

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Continuation model | Prefix occupies negative time and future births begin at zero | Time convention |
| Renderer evaluation | Covariance and P1 color gradients transform with coordinates | Affine tensor rules |
| Stable IDs | Transform changes parameters but never row order | Reordering rows |

## Notes

- Covariance uses congruence under the affine map and is returned through the canonical matrix-log
  representation. P1 gradients use the inverse coordinate Jacobian.
