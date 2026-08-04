# `render_prior_samples.py`

## Purpose

Turns latent-prior metrics into inspectable held-out video artifacts. Each animation compares the
fitted target, deterministic tokenizer upper path, and an independently sampled conditional prior.

## Components

### `main`
- **Does**: Restores prior/cache/tokenizer provenance, selects source-balanced held-out windows,
  samples and decodes latents, renders matched frames, and writes labeled comparison GIFs plus JSON.
- **Interacts with**: `latent_data.py`, `latent_prior.py`, `autoencoder.py`, and the production
  additive renderer.

### `frame_points`
- **Does**: Builds production-order normalized coordinates for selected frames without allocating the
  full spacetime grid.

### `_render`
- **Does**: Converts canonical features through the production field adapter and kNN renderer.
- **Rationale**: Visual artifacts should use the existing deployment-shaped renderer while exact
  sampled metrics continue to audit its approximation separately.

### `_panel` / `_row`
- **Does**: Convert rendered tensors into labeled, consistently ordered PIL comparison frames.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Research review | GIF columns stay target / tokenizer / prior under identical coordinates | Panel order |
| Production bridge | Canonical 22-D features convert through `features_to_field` | Feature layout |

## Notes

- Animations subsample time for research speed; the manifest records exact source frame indices.
- Image-CLIP conditioning is used here. Prompt-text sampling needs the same CLIP model followed by
  `LatentCache.normalize_condition`.
