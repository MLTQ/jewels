# `streaming.py`

## Purpose

Defines the persistent-state contract for unbounded jewel video. A monolithic fitted field receives
stable global IDs, finite-support lifecycles, and rolling windows that clamp a prefix, carry active
jewels, and own only genuinely new births.

## Components

### `JewelLifecycles`

- **Does**: stores continuous support bounds and discrete first/last/active-frame measurements
- **Interacts with**: temporal covariance from `splat_density.py`
- **Rationale**: visual active density, emission rate, and lifespan are separate quantities

### `RollingWindow`

- **Does**: identifies context, carried jewels, births, and the complete active commit set
- **Rationale**: overlap is immutable model context, not a second reconstruction to blend

### `frame_times` / `normalized_time_to_frame`

- **Does**: converts the fitted monolithic `[-1,1]` time axis to physical frame units
- **Rationale**: window-local normalization must not redefine lifespan or birth rate

### `measure_jewel_lifecycles`

- **Does**: assigns row-stable global IDs and measures each jewel's finite temporal support
- **Interacts with**: `temporal_standard_deviation` in `splat_density.py`

### `build_rolling_windows`

- **Does**: produces prefix/future training views and verifies carried IDs plus births exactly
  partition every committed active set

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `streaming_metrics.py` | Frame bounds are half-open and IDs index the original feature rows | ID or interval semantics |
| Continuation training | Context is clamped; only `birth_ids` may be newly emitted | Window ownership policy |
| Editor | Carried/protected jewel IDs survive every window transition | Regenerating IDs |

## Notes

- The current fitted checkpoint still stores normalized time. Physical frame values are derived from
  one monolithic fit; a later generative checkpoint should store time origin and frame period.
- Support is explicit and finite. This is what makes inactive-jewel omission exact for the truncated
  reference renderer.
