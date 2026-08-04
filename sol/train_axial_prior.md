# `train_axial_prior.py`

## Purpose

Trains the first scalable text-conditioned prior over the visually validated 16³ block hierarchy on
the allocated 8 GB GPU.

## Components

### `main`
- **Does**: Loads the leakage-safe coarse cache, trains fp16 axial flow with condition dropout and
  EMA, evaluates fixed held-out paths/distribution baselines, and writes resumable checkpoints.
- **Interacts with**: `axial_prior.py`, `latent_prior.flow_matching_loss`, `latent_data.py`, and
  `prior_evaluation.py`.

### `_atomic_checkpoint` / `_append_json`
- **Does**: Preserve resumable state and append-only experiment metrics.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Hierarchical sampler/editor | EMA checkpoint carries axial architecture, cache, and grid metadata | Checkpoint schema |
| Research comparison | Evaluation uses the existing fixed-path, retrieval, mean, and energy metrics | Protocol |
| RTX 2070S | Axial length is 16; no 4,096² score tensor is materialized | Attention layout |

## Notes

- The first run is a distribution/conditioning feasibility gate; rendered generation follows only
  if it beats retrieval and scene-mean baselines.
