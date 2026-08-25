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
- **Does**: Verifies the separately exposed residual is zero under the bounded contract and carries
  the declared unconstrained contribution under the residual contract.
- **Does**: Proves the native 7-point sampler sees independent spatial/temporal neighbors, residual
  checkpoints expand bitwise-exactly into a zero local adapter, and adapter-only freezing leaves no
  other trainable parameter.
- **Does**: Proves collapsed neighborhoods yield zero derivative features and that a bias-free
  derivative adapter cannot emit a residual even with nonzero learned weights at radius zero.
- **Does**: Verifies the positive derivative feature scale is checkpointed while preserving the
  zero-radius invariant.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Factorized-v3 gate | Appearance and geometry gradients are independent | Branch ownership |
| v2 causal control | Transplanted geometry is numerically identical | Channel mapping |
| Residual appearance gate | Expansion is exact before optimization and genuinely removes bounds | Contract behavior |
| Local appearance gate | Native evidence is continuous and base ownership remains exact | Sampler order, adapter names, or freeze semantics |
| Forced-evidence gate | No local difference means exactly no derivative residual | Bias or derivative semantics |
