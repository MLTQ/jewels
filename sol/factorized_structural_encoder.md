# `factorized_structural_encoder.py`

## Purpose

Defines v3 of the irregular video-to-jewel encoder. Geometry and appearance have disjoint
parameters so appearance optimization can no longer regularize mobile centres back toward uniform
coverage, while old v2 geometry can be transplanted exactly for causal continuations.

## Components

### `spacetime_trunk`
- **Does**: Builds the v2-compatible geometry feature trunk at the declared cell grid.
- **Rationale**: Exact layer compatibility makes a successful v2 geometry checkpoint reusable
  without importing its coupled colour rows.

### `sample_feature_volume`
- **Does**: Samples a feature volume at normalized continuous x/y/time centres.
- **Rationale**: Appearance follows predicted jewels rather than a fixed RGB slot lattice.

### `FactorizedStructuralJewelEncoder`
- **Does**: Emits centre, quaternion/scale, and opacity from the geometry branch.
- **Does**: Samples independent fine/coarse appearance volumes at detached centres, then predicts
  colour residuals and bounded colour gradients over continuous video seeds.
- **Does**: Freezes geometry by ordinary parameter ownership and exports the canonical 22-D audit
  layout.

### `load_v2_geometry`
- **Does**: Copies the compatible v2 trunk and channels 0–9 plus opacity channel 22 into the compact
  11-channel geometry head.
- **Rationale**: This creates a matched appearance-only control at the already-passing step-2,000
  geometry state.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Structural distillation | Prediction keys match `StructuralJewelEncoder` | Output schema |
| Irregular-field audit | `canonical_features` returns 22 columns | Feature layout |
| Checkpoint loader | Architecture is `factorized_structural_jewel_encoder_v3` and `model_args` is complete | Metadata |
| Geometry transplant | v2 trunk topology and 23-channel-per-slot head order remain stable | v2 compatibility |
