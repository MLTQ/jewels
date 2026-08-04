# `test_cache_latents.py`

## Purpose

Protects architecture-aware restoration at the frozen tokenizer/latent-cache boundary.

## Components

### `CacheLatentsTests`
- **Does**: Builds minimal historical padded and dense sparse checkpoints and verifies that cache
  restoration selects the corresponding codec class and accepts its state dictionary.
- **Interacts with**: `cache_latents._restore_tokenizer`, `autoencoder.py`, and
  `sparse_autoencoder.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dense latent caching | Sparse checkpoint metadata restores `SparseJewelAutoencoder` | Architecture dispatch |
| Prior reproducibility | Historical structured checkpoints remain loadable | Default architecture fallback |
