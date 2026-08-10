# `test_birth_mark_flow.py`

## Purpose

Protects the bounded stochastic mark-generation gate chosen after the prompted washout audit.

## Components

### `BirthMarkFlowTests`
- **Does**: Verifies noisy set rasterization, flow gradients and sampling, and hard spatial/lifecycle
  projection into externally supplied topology, plus the optional aligned RGB-guide path.
- **Discrete lifecycle check**: Projection bounds are expressed in physical stride frames.
- **Boundary check**: Valid centers outside the visible frame remain in their clamped edge cells
  without being moved onto the image boundary.
- **Interacts with**: `birth_mark_flow.py`, `GridSpec`, and temporal covariance recovery.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Oracle-topology experiment | Mark flow trains and samples variable-sized target sets | Flow signature |
| Future topology head | Sample projection cannot migrate jewels between emitted cells | Projection behavior |
| Correlated generation | Noisy raster exposes local distribution statistics | Raster channels |
