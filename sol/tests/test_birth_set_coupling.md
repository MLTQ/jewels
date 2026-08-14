# `test_birth_set_coupling.py`

## Purpose

Protects the linear-memory learned set primitive used by the coupled jewel-birth spike.

## Components

### `BirthSetCouplingTests`

- **Does**: Verifies cell moments, exact zero-residual initialization, first-step gradient flow,
  row-permutation equivariance, and information transfer from a neighboring occupied cell.
- **Interacts with**: `birth_set_coupling.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Coupled mark flow | Set blocks preserve row order and shape | Output semantics |
| Base augmentation | Untrained block is bit-identical to identity | Initialization |
| Large initial windows | Coupling uses scatter/3D raster operations, not padded pairwise attention | Complexity |
