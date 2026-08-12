# `lifecycle_appearance_rollout.py`

## Purpose

Runs the autonomous two-stream control across initial generation and continuation windows. The
frozen selected field owns topology, causal row selection, stable IDs, and temporal state while a
second field may evolve spatial geometry and appearance.

## Components

### `LifecycleAppearanceWindowReport`

- **Does**: Audits normalized, local projected, and global lifecycle equality for one stride and
  records the magnitude plus mean/active scaffold gate of the surviving appearance residual.

### `LifecycleAppearanceRollout`

- **Does**: Couples ordinary `ScaffoldMarkRollout` fields for the frozen base and constrained
  appearance stream, with exact topology/ID/lifecycle properties and a serializable report.

### `rollout_lifecycle_appearance_marks`

- **Does**: Predicts topology only from frozen-base carry, rasterizes each stream's own appearance
  context over base-owned row IDs, samples matched marks, and appends both fields in lockstep.
- **Interacts with**: `lifecycle_appearance_flow.py`, `scaffold_mark_data.py`, topology realization,
  and the deterministic render gate.
- **Rationale**: Candidate opacity or covariance must not feed back into later counts or identity;
  otherwise a nominal appearance tune can silently change which jewels exist.
- **Coordinate contract**: Lifecycle dimensions are copied after standardized integration, local
  topology projection, and local-to-global covariance conversion. This protects the serialized
  canonical field despite the nonlinear covariance transform.
- **Residual-mask contract**: Every feature outside the selected appearance set is copied at those
  same three boundaries, so opacity/time-gradient screens cannot leak through state feedback.
- **Spatial-gate contract**: Optional cell weights are expanded to canonical ranks and reapplied
  after sampling, projection, and global conversion. A zero-strength jewel is bit-identical to base.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Deterministic four-class gate | Frozen base matches the selected ordinary rollout for one seed | RNG/order |
| Interactive editor | Both fields have identical append-only row IDs and count tensors | Topology ownership |
| Temporal stability gate | Canonical lifecycle dimensions are bit-identical at every boundary | Copy stages |
| Appearance continuation | Candidate context uses candidate features at base-owned context rows | Context ownership |
| Residual screening | Reports serialize the exact mutable feature indices | Dimension policy |
| Scaffold saliency | Cell weights align one-to-one with guide strides and topology cells | Gate shape |

## Notes

- This spike deliberately leaves topology conditioned on the frozen full-vector stream. It tests
  factorized ownership before spending compute on a purpose-built lifecycle network.
