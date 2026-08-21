# `test_distill_structural_encoder.py`

## Purpose

Protects the set-correspondence, tube-orientation, opacity-weighted placement, and differentiable
sparsity helpers used by structural teacher distillation.

## Components

### `DistillHelperTests` / `OrientationTests`
- **Does**: Verifies symmetric Chamfer coverage, teacher descriptor shapes/weights, and sign-invariant
  principal-axis matching.
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

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Irregular-field trainer | Density gradients reach centres and opacity weights | Loss semantics |
| Structural audits | Active means opacity above 2% | Threshold policy |
| Appearance-only continuation | Trunk and geometry/opacity head rows do not move | Freeze semantics |
