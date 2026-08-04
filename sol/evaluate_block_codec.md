# `evaluate_block_codec.py`

## Purpose

Evaluates whether the fixed 16³ block hierarchy preserves the grid-32 tokenizer's held-out rendered
detail, rather than accepting PCA variance as a proxy for video quality.

## Components

### `main`
- **Does**: Inverse-PCA decodes aligned coarse codes, restores fine tokenizer scale, sparse-decodes
  jewels, and reports exact sampled-render PSNR/count over source-balanced held-out windows.
- **Interacts with**: `block_codec.py`, both latent caches, `cache_latents._restore_tokenizer`, and
  `render.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Hierarchy gate | Fine/coarse caches have identical names and split order | Cache alignment |
| Prior comparison | Reports held-out fine-latent MSE plus the existing render metric schema | Output schema |
| Visual decoder | PCA reconstruction is denormalized with the parent fine cache | Scale/order semantics |
