# `test_cache_latents.py`

## Purpose

Protects architecture-aware restoration at the frozen tokenizer/latent-cache boundary.

## Components

### `CacheLatentsTests`
- **Does**: Builds minimal historical padded and dense sparse checkpoints and verifies that cache
  restoration selects the corresponding codec class and accepts its state dictionary.
- **Does**: Verifies training latents expand across training templates while a source-held-out latent
  receives only its unseen evaluation template, and rejects tokenizer/manifest split disagreement.
- **Interacts with**: `cache_latents._restore_tokenizer`, `autoencoder.py`, and
  `sparse_autoencoder.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dense latent caching | Sparse checkpoint metadata restores `SparseJewelAutoencoder` | Architecture dispatch |
| Prior reproducibility | Historical structured checkpoints remain loadable | Default architecture fallback |
| Prompt prior | Template expansion preserves exact source-level train/validation ownership | Split binding |
