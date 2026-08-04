# `fit_block_codec.py`

## Purpose

Fits the train-only fixed 2³ block hierarchy and creates the immutable coarse-cache interface for an
axial prior experiment.

## Components

### `main`
- **Does**: Fits `BlockPCACodec`, atomically saves it, encodes every fine latent field, computes
  train-only per-coarse-cell normalization, and saves a provenance-linked `LatentCache`.
- **Interacts with**: `block_codec.py` and `latent_data.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Axial prior trainer | Coarse cache uses the existing normalized latent/condition split API | Cache schema |
| Block evaluator | Metadata links parent cache, codec, fine/coarse shapes, and variance | Provenance fields |

## Notes

- The PCA is fit on already train-normalized fine latents and never sees held-out source blocks.
