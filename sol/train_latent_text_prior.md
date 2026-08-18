# `train_latent_text_prior.py`

## Purpose

Trains the text-conditioned latent flow on cached encoder latents and reports the
correct/shuffled/null text battery this project uses for every conditioning claim.

## Components

### `pack` / `unpack`
- **Does**: Concatenate per-cell features with flattened slot seeds into one token vector and
  invert it, so the flow sees a single `(cells, feature_dim)` tensor.

### `main`
- **Does**: Fits the standardizer on the train split only, trains with text dropout for
  classifier-free guidance, and evaluates held-out velocity error under correct, shuffled, and
  null prompts on a fixed noise/time path.
- **Rationale**: Fixed evaluation paths make the three arms differ only by conditioning.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Render gate | Checkpoint meta carries `model_args`, `standardizer`, and encoder provenance | Meta schema |

## Notes

- Latent MSE is a weak selectivity metric when most latent variance is instance-specific
  detail; the rendered gate is authoritative.
