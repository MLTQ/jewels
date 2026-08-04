# `evaluate_latent_prior.py`

## Purpose

Re-evaluates a saved EMA raster-flow checkpoint with a larger fixed held-out sampling budget without
resuming training.

## Components

### `main`
- **Does**: Restores the exact cache/model pair, samples held-out conditions, compares latent energy
  distance against scene-mean and CLIP-retrieval distributions, and writes a reproducible JSON report.
- **Interacts with**: `latent_data.py`, both prior architectures, and `prior_evaluation.py`.

### `_restore_prior`
- **Does**: Restores historical global-raster or hierarchical axial EMA weights from checkpoint
  architecture metadata.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Prior research record | Architecture, checkpoint/cache identity, seed, CFG, solver steps, names, and metrics | Output schema |

## Notes

- `--samples` is capped by the number of held-out windows.
- Keep seed, sample count, integration steps, and CFG fixed for model comparisons.
