# `render_dense_tokenizer.py`

## Purpose

Provides the decisive visual gate for the 45k-jewel sparse tokenizer: matched held-out fitted targets
and deterministic round-trips across time.

## Components

### `main`
- **Does**: Restores sparse codec/provenance, selects source-balanced validation windows, renders 16
  matched temporal positions, and writes labeled GIFs plus a manifest. Target panels report the
  actual jewel count from the loaded example rather than assuming a legacy corpus density.
- **Diagnostic selection**: `--names` renders exact corpus examples, including training windows,
  when separating codec memorization from held-out scene generalization. Without it, selection stays
  source-held-out and balanced.
- **Capacity diagnostic**: `--slots-override` can increase, never reduce, the checkpoint slot budget
  for an explicitly labeled cross-domain render. Weights, grid, normalization, and latent width stay
  frozen.
- **Interacts with**: `sparse_autoencoder.py` and production-render helpers in
  `render_prior_samples.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dense tokenizer selection | Panel order is fitted target / sparse round-trip | Visual semantics |
| Research record | Panel labels and manifest carry the loaded target count | Label/count provenance |
| Cross-domain audit | Manifest records trained and rendered slot capacities | Capacity provenance |

## Notes

- This uses production kNN rendering for inspectability; exact sampled PSNR remains the numerical
  correctness protocol.
