# `cache_latents.py`

## Purpose

Runs the selected tokenizer encoder once and creates the immutable training interface for latent
flow experiments. Conditions and provenance are aligned by fitted-window filename.

## Components

### `main`
- **Does**: Restores the tokenizer, encodes corpus batches, loads/unit-normalizes CLIP sidecars,
  computes train-source-only latent and condition statistics, and atomically saves `LatentCache`.
- **Interacts with**: `corpus.py`, both tokenizer implementations, and `latent_data.py`.

### `_restore_tokenizer`
- **Does**: Selects historical padded or sparse variable-count codec construction from checkpoint
  architecture metadata and restores the exact weights.
- **Rationale**: Dense grid-32 latents must use the selected sparse encoder without breaking earlier
  small-raster cache reproducibility.

### `_sha256`
- **Does**: Fingerprints the exact tokenizer checkpoint that produced the cache.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train_latent_prior.py` | Cache has frozen normalized latents and aligned conditions | Cache schema |
| Reproducibility audit | Tokenizer hash, architecture, step, corpus, grid, and CLIP identity travel with cache | Provenance fields |

## Notes

- The split is restored from the tokenizer checkpoint, not randomly recreated.
- Sidecars were produced from CLIP image embeddings. Text embeddings share the vector space but have
  a modality gap; this cache proves conditioning mechanics, not broad prompt understanding.
