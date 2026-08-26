# `hierarchical_jewel_decoder.py`

## Purpose

Defines the learned bridge from a hierarchical product-code phrase to continuous Jewels. The model
receives only emitted role tokens, coarse cell, exact irregular pair anchor, and count. It predicts
continuous corrections to the frozen token prototypes; it never receives the target residual at
inference.

## Components

### `HierarchicalPhraseBatch` / `build_hierarchical_phrase_batch`
- **Does**: Aligns pair layout/covariance tokens with individual surface/gradient tokens and the
  exact pair-bundle target. Padding is explicit for one-Jewel terminal pairs.
- **Rationale**: The batch is exactly the Gate-0d casting phrase, not unused oracle roles.

### `build_sampled_hierarchical_phrase_batch`
- **Does**: Samples pair bundles per training source before token assignment and assigns only the
  four active hierarchical roles.
- **Rationale**: Validation remains exhaustive, while training is source-balanced and avoids
  spending most compute on unused factor assignments.

### `concatenate_phrase_batches` / `residual_scale`
- **Does**: Builds train-owned datasets and robust per-row/per-feature correction scales.

### `HierarchicalPhraseDecoder`
- **Does**: Embeds six active product-code positions, the cell, count, and Fourier anchor; an MLP
  predicts a continuous 2x22 correction around the frozen prototype composition. Corrections are
  smoothly bounded to three train-owned RMS units per feature to prevent unsupported splats.
- **Rationale**: Product tokens capture reusable language; the learned head restores correlations
  that independent role codebooks deliberately omit.

### `phrase_decoder_loss` / `phrase_values_to_features`
- **Does**: Trains on masked normalized correction error and restores canonical render features.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Decoder trainer | Token columns are pair layout/covariance, surface rows 0/1, gradient rows 0/1 | Phrase schema |
| Scientific gate | Forward input excludes `target_values` and exact residual | Model input contract |
| Renderer | Predicted values decode through pair anchors and train-owned normalizer | Feature contract |
