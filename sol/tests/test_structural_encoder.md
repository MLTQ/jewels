# `test_structural_encoder.py`

## Purpose

Protects the scarce tube-capable encoder's geometry, rendering, colour-seeding, and gradient
contracts.

## Components

### `StructuralEncoderTests`
- **Does**: Verifies quaternion rotations and expressive precision factors.
- **Does**: Matches canonical rendering and propagates gradients through shape parameters.
- **Does**: Matches the support-tiled renderer to an explicit five-sigma value/gradient oracle for
  arbitrary rotated precision factors.
- **Does**: Ensures arbitrary non-cubic slot counts stay strictly inside their proposal cell.
- **Does**: Confirms optional colour seeds follow predicted continuous centres rather than a fixed
  raster location.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Structural trainers | Canonical shapes and differentiable render equivalence | Prediction schema |
| Irregular-field gate | 36-slot layouts cannot overflow their cells | Proposal layout |
