# `test_block_codec.py`

## Purpose

Protects the fixed block hierarchy's canonical inverse and persisted state contract.

## Components

### `BlockCodecTests`
- **Does**: Verifies an unreduced PCA basis reconstructs fine latents numerically and serialized
  codec state preserves coarse codes and shape.
- **Interacts with**: `block_codec.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Fine decoder | Block encode/decode never permutes raster cells | Inverse layout |
| Coarse cache | Restored codec produces identical code tensors | State fields |
