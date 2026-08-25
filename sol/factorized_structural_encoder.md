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
- **Does**: Supports the legacy `bounded` appearance contract and an explicit `residual` contract
  that adds unconstrained RGB/Jacobian residuals after the bounded prediction.
- **Does**: Supports a zero-expanded `local_adapter` contract that samples a native-resolution
  7-point cross around each detached irregular jewel center and adds a separately owned residual.
- **Does**: Supports a forced-evidence `derivative_adapter` whose bias-free input is only central
  RGB x/y/time differences plus six-neighbor contrast; a zero-radius input must remain zero.
- **Does**: Exposes the 12-D residual contribution separately as `appearance_residual` so training
  and audits can measure color and Jacobian energy without reverse-engineering the composed output.
- **Rationale**: Fitted responsibility targets frequently exceed sigmoid RGB and bounded-gradient
  ranges; a zero-initialized residual tests that bottleneck without changing the source render.

### `load_bounded_appearance_expansion`
- **Does**: Loads every parameter from a bounded v3 checkpoint into a residual model while requiring
  that the only missing state is the zero-initialized residual head.
- **Rationale**: The expanded continuation starts exactly at its matched source rather than changing
  predictions through a parameterization reinterpretation.

### `sample_native_neighborhood` / `load_local_adapter_expansion`
- **Does**: Samples center, spatial +/-x/y, and temporal +/-t RGB at continuous jewel locations;
  expands a residual checkpoint while requiring that only the zero local adapter is absent.
- **Rationale**: The old fine branch is stride-2 and receives just one native RGB sample, so it can
  minimize average error without observing a boundary around the jewel. Exact expansion isolates
  the value of local evidence from any base-checkpoint change.

### `native_neighborhood_derivatives`
- **Does**: Converts the ordered cross into nine signed channel/axis differences and three local
  contrast values for a bias-free adapter.
- **Rationale**: The raw local adapter tied its radius-0 control under both render and LPIPS losses,
  showing that generic base features/capacity dominated. The derivative contract makes any nonzero
  content-dependent output causally depend on native irregular-neighborhood evidence.
- **Calibration**: The first forced-evidence screen measured gradients about 30x below the stable
  generic adapter. The trainer explicitly defaults new derivative runs to a checkpointed scale 32;
  the model constructor falls back to 1 so pre-scale checkpoints retain their original semantics.

### `freeze_base_for_local_adapter`
- **Does**: Freezes geometry, the proven residual appearance base, and background while leaving only
  the local adapter trainable.
- **Rationale**: A capacity/input/perceptual screen must not silently spend its gain by rewriting the
  already replicated 20 dB solution.

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
| Appearance expansion | `appearance_contract` is checkpointed; missing legacy values mean `bounded` | Contract semantics |
| Appearance diagnostics | `appearance_residual` is zero for bounded models and `N x 12` for every model | Prediction dictionary |
| Local-adapter diagnostics | `appearance_adapter_residual` is always `N x 12`; non-adapter contracts return exact zeros | Prediction dictionary |
| Geometry transplant | v2 trunk topology and 23-channel-per-slot head order remain stable | v2 compatibility |
| Residual-to-local expansion | Every residual tensor/output is bitwise preserved before training | Adapter module names or defaults |
| Forced local evidence | Zero neighborhood differences produce exactly zero derivative-adapter output | Sampler order, biases, or derivative features |
| Derivative calibration | Positive scale is explicit in `model_args`; zero evidence remains zero | Scale default or metadata |
