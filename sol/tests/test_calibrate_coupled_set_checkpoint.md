# `test_calibrate_coupled_set_checkpoint.py`

## Purpose

Protects inference-only calibration of the coupled jewel-birth set residual.

## Components

### `CalibrateCoupledSetCheckpointTests`

- **Does**: Verifies that only the final set residual projection is scaled, base and set-encoder
  weights stay exact, provenance is copied without mutating the source, optimizer state is removed,
  and repeated calibration is rejected.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Calibration screen | Strength zero/base and one/candidate endpoints remain meaningful | State selection |
| Reproducibility | Source checkpoint and scalar strength remain explicit | Metadata |
