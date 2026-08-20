# `test_stprim_render.py`

## Purpose

Protects the renderer used by stage-1 fitting from silently reverting to center-distance KNN as a
claimed correctness approximation.

## Components

### `elongated_counterexample()`
- Builds 65 narrow, nearer zero-color splats and one farther bright splat elongated diagonally
  through space and normalized time.
- The bright splat contributes more than 0.8 at the query but is excluded by 64-nearest-center KNN.

### `ProductionRenderTests`
- Verifies support mode matches the all-primitive oracle on the counterexample.
- Verifies an insufficient conservative candidate capacity raises `SupportOverflowError`.
- Verifies support selection remains differentiable with respect to selected primitive parameters.
- Verifies axis-aligned and isotropic causal controls project exactly, while invalid controls fail.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `stprim/models/render.py` | Support mode is complete within declared finite support and never silently overflows | Culling semantics or overflow behavior |
