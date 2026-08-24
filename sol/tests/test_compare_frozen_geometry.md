# `test_compare_frozen_geometry.py`

## Purpose

Protects the direct state-tensor proof used by frozen-appearance replication.

## Components

### `FrozenGeometryComparisonTests`

- **Does**: Confirms appearance-only changes are ignored, a one-ULP geometry change is detected, and
  missing factorized geometry ownership is rejected.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `compare_frozen_geometry.py` | Equality is bitwise and limited to geometry owners | Tensor selection |
