# `test_factorized_structural_encoder.py`

## Purpose

Protects the architectural separation and checkpoint-compatibility claims of the v3 irregular
encoder.

## Components

### `FactorizedStructuralEncoderTests`
- **Does**: Verifies prediction and canonical feature shapes.
- **Does**: Proves an appearance-only loss has no gradient path into geometry.
- **Does**: Proves the v2 trunk/row transplant reproduces centre, covariance inputs, and opacity
  exactly.
- **Does**: Verifies explicit geometry freeze leaves the appearance MLP trainable.
- **Does**: Proves bounded-to-residual expansion preserves every initial prediction exactly.
- **Does**: Proves the residual contract can emit RGB and Jacobians outside the legacy bounds.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Factorized-v3 gate | Appearance and geometry gradients are independent | Branch ownership |
| v2 causal control | Transplanted geometry is numerically identical | Channel mapping |
| Residual appearance gate | Expansion is exact before optimization and genuinely removes bounds | Contract behavior |
