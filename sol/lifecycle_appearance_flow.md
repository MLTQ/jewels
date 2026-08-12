# `lifecycle_appearance_flow.py`

## Purpose

Defines the two-stream sampling control that separates persistent jewel lifecycle from editable
spatial geometry and appearance. The selected mark flow remains frozen and is integrated
independently; a candidate flow shares its initial noise but cannot change temporal state.

## Components

### `LIFECYCLE_DIMENSIONS` / `SPATIAL_APPEARANCE_DIMENSIONS`

- **Does**: Partitions the canonical 22-D mark into temporal center/time-coupled log covariance
  `(2,5,7,8)` and every remaining spatial/appearance dimension.
- **Rationale**: Soft loss weighting did not prevent quiet-region temporal regression.

### `APPEARANCE_DIMENSION_SETS`

- **Does**: Names reproducible residual masks for all appearance, static spatial detail,
  color/detail only, color only, geometry only, no time-gradient, and no-opacity controls.
- **Rationale**: Even with exact jewel lifecycles, opacity and RGB time-gradients can introduce
  rendered flicker; dimension controls localize that failure before another training run.

### `copy_lifecycle_dimensions`

- **Does**: Copies lifecycle coordinates from a reference tensor into a cloned candidate tensor.
- **Rationale**: Exact assignment, rather than a penalty, makes temporal ownership testable.

### `constrain_appearance_dimensions`

- **Does**: Restores every coordinate outside an explicit mutable residual set from the frozen
  reference after an integration or coordinate-transform step, and optionally blends selected
  coordinates per mark with bounded scaffold-owned strengths.
- **Rationale**: Hard dimension masks localize feature coupling; per-cell gates can keep residuals
  away from quiet scaffold regions without target masks or semantic labels at inference.

### `LifecycleLockedSample`

- **Does**: Returns both standardized trajectories plus exact lifecycle and residual-size audits.

### `sample_lifecycle_locked_birth_marks`

- **Does**: Draws one shared Gaussian initial state, Euler-integrates frozen and candidate flows
  with their respective contexts, and restores every non-selected coordinate after every step.
- **Interacts with**: `BirthMarkFlowModel` in `birth_mark_flow.py` and the paired autonomous rollout.
- **Rationale**: Interleaving matched trajectories isolates appearance changes without replacing
  the selected model's topology/lifetime dynamics or introducing a second random draw.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Lifecycle/appearance rollout | Base sampling matches ordinary Euler sampling for the same seed | RNG or step order |
| Stable jewel editor | Candidate frozen coordinates and zero-strength rows are bit-identical | Dimension/gate policy |
| Matched checkpoint gate | Both models share feature, guide, text, and topology contracts | Compatibility checks |

## Notes

- The default `all` mask leaves RGB time gradients and opacity appearance-owned; stricter named
  masks test whether either causes temporally unstable color despite exact geometric lifecycle.
- The base stream is intentionally a full frozen mark model in this spike. A later purpose-built
  lifecycle head can replace it only after this ownership split passes the visual gate.
