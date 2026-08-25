# `perceptual_objective.py`

## Purpose

Provides direct train-only feature-perceptual supervision for frozen-geometry appearance
experiments. The network is loaded lazily so ordinary development and the core test suite do not
require LPIPS weights.

## Components

### `build_lpips_training_metric`
- **Does**: Builds an evaluation-mode LPIPS network and freezes its parameters without disabling
  gradients with respect to rendered input pixels.
- **Rationale**: The perceptual network is a fixed measuring instrument, not another trainable arm.

### `perceptual_training_loss`
- **Does**: Validates matching `F,H,W,3` videos, applies the display clamp, converts to LPIPS channel
  order, and returns the mean differentiable feature distance.
- **Rationale**: Training must optimize the same displayed image domain used by held-out LPIPS;
  separate range penalties remain responsible for values outside `[0,1]`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Structural distillation | Scalar loss backpropagates into rendered pixels but not LPIPS weights | Gradient or clamp semantics |
| Core tests | Metric is injectable and `lpips` is imported only by the factory | Eager dependency loading |
| Frozen appearance gate | Only training-source frames reach this objective | Caller data ownership |
