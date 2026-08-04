# `analyze_latent_hierarchy.py`

## Purpose

Runs the frozen grid-32 latent redundancy audit and writes its provenance-linked JSON record.

## Components

### `main`
- **Does**: Loads only training-source normalized latents, restores the cached grid shape, computes
  hierarchy diagnostics, and records the tokenizer fingerprint.
- **Interacts with**: `latent_data.py` and `latent_hierarchy.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Prior design record | JSON carries cache path, tokenizer hash, grid, correlations, pooling, and PCA | Report schema |
| Leakage control | Diagnostics use only `train_mask=True` samples | Split selection |
