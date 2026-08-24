# `test_appearance_objective.py`

## Purpose

Protects the calibrated full-frame appearance objective used after irregular geometry is frozen.

## Components

### `AppearanceObjectiveTests`

- **Does**: Locks the legacy multiscale wrapper, matching-video floor, temporal gradient path,
  display-range diagnostics, separated residual energies, and invalid-weight checks.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `appearance_objective.py` | Every named term is finite and differentiable where applicable | Loss semantics |
| Existing distillation tests | Compatibility wrapper retains its registered default | Default weighting |
