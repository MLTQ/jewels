# `train_latent_prior.py`

## Purpose

Trains the first text-conditioned rectified-flow model over the frozen editable raster latents.
Defaults are intentionally small enough for the allocated 8 GB RTX 2070 SUPER.

## Components

### `main`
- **Does**: Loads the leakage-safe cache, trains with fp16 flow matching and condition dropout,
  maintains EMA weights, evaluates fixed held-out paths/baselines, and writes resumable checkpoints.
- **Interacts with**: `latent_data.py`, `latent_prior.py`, and `prior_evaluation.py`.

### `_atomic_checkpoint` / `_append_json`
- **Does**: Preserve interruption-safe model state and an append-only metric curve.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Future sampler/editor | Checkpoint contains EMA prior, model dimensions, and cache provenance | Checkpoint schema |
| Research comparison | Step zero and periodic evaluations use the fixed prior protocol | Log/eval schema |
| 2070S experiment | Defaults fit 8 GB using `cuda:1` and fp16 scaling | Memory-affecting defaults |

## Notes

- Training conditions are CLIP image embeddings. Text-tower prompting is architecturally compatible,
  but the current single-domain corpus cannot prove broad language grounding.
- EMA, optimizer, and scaler states are all resumable; the cache fingerprints the frozen tokenizer.
