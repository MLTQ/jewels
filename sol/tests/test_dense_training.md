# `test_dense_training.py`

## Purpose

Protects the foreground-aware dense-tokenizer training objective.

## Components

### `DenseTrainingTests`
- **Does**: Verifies that mixed uniform/motion-importance point selection produces a finite render
  loss and propagates finite gradients into decoded jewel features.
- **Does**: Checks balanced per-frame source-video pools, normalized coordinates, and the
  precomputed-pool render-loss path.
- **Interacts with**: `train_dense_autoencoder._sampled_render_loss`, the exact renderer, and compact
  raster targets.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dense tokenizer experiment | Motion sampling remains differentiable through predictions | Render-loss gradient path |
| Whole-scene fidelity | Uniform and motion points coexist when the fraction is between zero and one | Sampling semantics |
