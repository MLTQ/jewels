# `test_dense_training.py`

## Purpose

Protects the foreground-aware dense-tokenizer training objective.

## Components

### `DenseTrainingTests`
- **Does**: Verifies that mixed uniform/motion-importance point selection produces a finite render
  loss and propagates finite gradients into decoded jewel features.
- **Does**: Checks balanced per-frame source-video pools, normalized coordinates, and the
  precomputed-pool render-loss path.
- **Does**: Verifies each frame reserves disjoint motion and saturated-chroma samples, including a
  static rare-color pixel that a temporal score alone would ignore and grayscale motion controls
  that cannot win the chroma quota.
- **Interacts with**: `train_dense_autoencoder._sampled_render_loss`, the exact renderer, and compact
  raster targets.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dense tokenizer experiment | Saliency sampling remains differentiable through predictions | Render-loss gradient path |
| Whole-scene fidelity | Uniform and source-saliency points coexist at intermediate fractions | Sampling semantics |
