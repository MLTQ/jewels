# `test_calibrate_responsibility_distillation.py`

## Purpose

Protects the parameter-group gradient norm used to preregister responsibility-loss weights.

## Components

### `ResponsibilityCalibrationTests`
- **Does**: Verifies the joint norm includes used parameters and safely ignores unused parameters.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Calibration report | Gradient norm is the L2 norm across the declared parameter group | Norm semantics |
