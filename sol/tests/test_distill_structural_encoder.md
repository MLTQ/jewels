# `test_distill_structural_encoder.py`

## Purpose

Protects the set-correspondence, tube-orientation, opacity-weighted placement, and differentiable
sparsity helpers used by structural teacher distillation. Architecture separation is covered by
`test_factorized_structural_encoder.py`, local correspondence by
`test_local_teacher_distillation.py`; this file retains the v2 mixed-head freeze contract.
Renderer-responsibility moment math and detached targets are likewise isolated there.

## Components

### `DistillHelperTests` / `OrientationTests`
- **Does**: Verifies symmetric Chamfer coverage, teacher descriptor shapes/weights, and sign-invariant
  principal-axis matching, including the absolute mean-log-scale target.
- **Does**: Verifies mixed tilt is zero for pure space/time axes and one at a 45-degree trajectory.

### `DensityMatchingTests` / `ValidationSelectionTests`
- **Does**: Verifies normalized soft occupancy, opacity-weighted allocation, density gradients, and
  active-fraction behavior around the canonical 2% opacity threshold.
- **Does**: Verifies bounded evaluation preserves explicit source order and rejects ownership errors.
- **Does**: Verifies the shared delayed structural schedule is exactly zero through its start step,
  ramps linearly, and saturates at one; density and sparsity use independent instances.

### `FrozenGeometryTests`
- **Does**: Verifies appearance-only continuation freezes the shared trunk, masks only geometry and
  opacity rows, restores those rows exactly after an optimizer-like mutation, and leaves appearance
  rows trainable.

### `AppearanceImageLossTests`
- **Does**: Locks the near-zero matching-image floor and verifies colour/edge disagreement produces
  a larger differentiable multiscale loss.
- **Does**: Verifies deterministic contiguous-frame selection and rejects a frame request longer
  than its source video.
- **Does**: Proves the frozen-geometry report distinguishes exact equality from a one-ULP change.
- **Does**: Proves the all-base-parameter audit distinguishes exact equality from a one-ULP change.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Irregular-field trainer | Density gradients reach centres and opacity weights | Loss semantics |
| Structural audits | Active means opacity above 2% | Threshold policy |
| Appearance-only continuation | Trunk and geometry/opacity head rows do not move | Freeze semantics |
| Factorized appearance training | Full-image multiscale/edge loss is differentiable | Objective semantics |
| Frozen geometry audit | Equality is bitwise, not tolerance-based | Gate semantics |
| Frozen adapter base audit | Every non-adapter parameter is bitwise source-equal | Optimizer ownership |
