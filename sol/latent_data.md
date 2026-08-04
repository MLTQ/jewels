# `latent_data.py`

## Purpose

Defines the frozen interface between the proven jewel tokenizer and generative-prior experiments.
It keeps normalization and source-level leakage boundaries inseparable from the cached latents.

## Components

### `LatentCache`
- **Does**: Holds raw raster latents, unit CLIP conditions, names, source IDs, split mask, train-only
  per-cell latent normalization, per-dimension condition whitening, and provenance metadata.
- **Rationale**: Cell-wise normalization removes the frozen encoder's positional offsets while using
  only training sources for statistics. CLIP whitening removes the dominant shared scene component
  so subtle window differences are visible to the conditioner.

### `LatentCache.split` / `denormalize` / `normalize_condition`
- **Does**: Returns normalized train or validation tensors and maps generated tensors back to the
  tokenizer's latent space. Applies the same unit-normalize-and-whiten transform to future text CLIP
  vectors.

### `save_latent_cache` / `load_latent_cache`
- **Does**: Atomically persist and validate the cache contract.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Prior trainer | Normalized latents and train-whitened 512-D conditions | Tensor schema |
| Prior evaluator | Exact names, sources, split, normalization, tokenizer provenance | Metadata schema |
| Tokenizer decoder | `denormalize` restores encoder-scale raster latents | Normalization axes |

## Notes

- The cache contains latents, not source jewels, so the prior loop never reruns the expensive encoder.
- Validation rejects any source ID appearing on both sides of the split.
- Pre-whitening spike caches load with identity condition statistics so their checkpoints remain
  evaluable under later metric protocols.
