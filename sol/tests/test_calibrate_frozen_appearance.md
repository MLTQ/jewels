# `test_calibrate_frozen_appearance.py`

## Purpose

Protects the no-mutation gradient measurement used to register frozen-appearance loss weights.

## Components

### `FrozenAppearanceCalibrationTests`

- **Does**: Verifies the analytical joint L2 norm, confirms `.grad` remains empty, and permits
  parameters unused by one named component.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `calibrate_frozen_appearance.py` | Calibration measures rather than accumulates gradients | Autograd behavior |
