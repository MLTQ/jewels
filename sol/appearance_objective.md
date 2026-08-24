# `appearance_objective.py`

## Purpose

Defines the full-frame appearance objective used to sharpen a frozen irregular jewel field without
changing its centers, covariance, or opacity. Named components keep calibration and negative results
auditable instead of collapsing them into one unexplained perceptual weight.

## Components

### `AppearanceObjective` / `appearance_objective`

- **Does**: Combines multiscale RGB Charbonnier, spatial edges, contiguous-time differences, global
  spatiotemporal SSIM, and display-range excess with explicit non-negative weights.
- **Rationale**: Independent voxel MSE misses coherent edges and temporal glitter; named terms allow
  gradient-scale calibration before optimization.

### `multiscale_image_loss`

- **Does**: Compatibility wrapper for the original RGB-pyramid plus spatial-edge objective.
- **Contract**: Its numerical definition remains `rgb + 0.5 * mean(horizontal, vertical)`.

### `range_excess_loss` / `range_diagnostics`

- **Does**: Measures and optionally penalizes rendered RGB outside `[0,1]` without clipping the
  residual jewel parameterization.
- **Rationale**: Rare negative residuals appear as dark speckles after display clamping.

### `residual_energy`

- **Does**: Separates mean-square residual RGB from residual RGB-Jacobian energy.
- **Rationale**: The two parameter groups have different physical scales and must not share an
  opaque regularizer.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Structural distillation | Inputs use contiguous `F,H,W,3` video order | Shape or time semantics |
| Frozen-appearance calibration | Every component remains separately accessible | Dataclass fields |
| Legacy appearance-grid runs | `multiscale_image_loss` preserves its old value | Default weights |
