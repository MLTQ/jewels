# `latent_text_prior.py`

## Purpose

Stage 1's generator: a rectified flow over the frozen encoder's cell-token latent, conditioned
on prompt **token sequences** via cross-attention. Generation happens in a structured latent
with a proven decoder, so joint composition is the decoder's job — the failure mode that
sank direct mark-space sampling.

## Components

### `LatentStandardizer`
- **Does**: Per-channel mean/std fitted on the training split so the flow sees unit-scale
  targets; `denormalize` returns generated samples to encoder units.

### `_Block`
- **Does**: Self-attention over the 2,048 cell tokens, cross-attention to text tokens, then an
  MLP — each gated by a zero-initialized, time-conditioned modulation, so the model starts as
  an exact identity and learns how much conditioning to admit.

### `LatentTextPrior`
- **Does**: Predicts flow velocity from a noised latent, flow time, and prompt tokens; the
  output projection is zero-initialized (zero velocity at step 0). `sample` Euler-integrates
  from noise with optional classifier-free guidance against the learned null-text embedding.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train_latent_text_prior.py` | `(B, n_cells, cell_dim+seed_dim)` latent packing | Packing order |
| `evaluate_latent_text_prior.py` | `sample()` returns standardized latents to denormalize | Sampling API |
| Decode path | Unpacked `cells`/`seed` match the encoder's `decode` contract | Latent layout |

## Notes

- Token-sequence conditioning is deliberate: pooled sentence vectors are what earlier
  CLIP-conditioned attempts failed with.
