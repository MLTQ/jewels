# `test_frontier_contribution_loss.py`

## Purpose

Protects the targeted differentiable loss used to remove weak-support-tail ramp-in at generated
window frontiers.

## Components

### `FrontierContributionLossTests`

- **Does**: Verifies frontier-aligned jewels have greater peak alpha, identical marks score zero,
  and temporal misalignment yields finite nonzero center/opacity gradients across the detached
  covariance-eigen boundary.
- **Interacts with**: `frontier_contribution_loss.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Fine-tune gate | Loss differentiates into temporal center/covariance/opacity | Gradient path |
| Calibration | Identical target-topology rows have a zero objective | Baseline |
