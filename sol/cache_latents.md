# `cache_latents.py`

## Purpose

Runs the selected tokenizer encoder once and creates the immutable training interface for latent
flow experiments. It supports historical image-CLIP sidecars and leakage-safe expansion across
multiple text prompt templates.

## Components

### `main`
- **Does**: Restores the tokenizer, encodes one or more corpus roots, binds either image sidecars or
  a manifest-validated prompt cache, computes train-source-only statistics, and saves `LatentCache`.
- **Interacts with**: `corpus.py`, both tokenizer implementations, and `latent_data.py`.

### `_restore_tokenizer`
- **Does**: Selects historical padded or sparse variable-count codec construction from checkpoint
  architecture metadata and restores the exact weights.
- **Rationale**: Dense grid-32 latents must use the selected sparse encoder without breaking earlier
  small-raster cache reproducibility.

### `_sha256`
- **Does**: Fingerprints the exact tokenizer checkpoint that produced the cache.

### `_expand_prompt_conditioned_latents`
- **Does**: Repeats each training latent across its training prompt templates and binds each held-out
  latent only to its unseen evaluation template.
- **Interacts with**: Typed `FittedExample` source IDs, `PromptEmbeddingCache`, and the immutable
  prompt manifest.
- **Rationale**: The prior sees linguistic variation without allowing a validation phrase or source
  to enter training; exact video-stem matching prevents class-only attachment mistakes.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train_latent_prior.py` | Cache has frozen normalized latents and aligned conditions | Cache schema |
| Reproducibility audit | Tokenizer hash, architecture, step, corpus, grid, and CLIP identity travel with cache | Provenance fields |

## Notes

- The split is restored from the tokenizer checkpoint, not randomly recreated.
- Prompt mode requires both `--prompt-cache` and `--prompt-manifest`; their digests must agree.
- Multiple corpus roots are read in place; the 120k-jewel checkpoints are not physically copied.
- Prompt mode produces 36 training rows and four held-out rows for the current 12/4 corpus, while
  retaining source IDs so repeated prompt templates cannot cross the split boundary.
- Image sidecars remain supported for historical experiments.
