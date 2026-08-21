# `test_compare_field_structure.py`

## Purpose

Protects the mixed-spacetime orientation metric used to reject elongated but non-trajectory
jewels.

## Components

### `FieldStructureTests`
- **Does**: Constructs a known 45-degree spacetime covariance and verifies mixed tilt is near one.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Structural gate | Diagonal spacetime axes score high; pure axes score zero | Mixed-tilt formula |
