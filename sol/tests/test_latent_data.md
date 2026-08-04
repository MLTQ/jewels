# `test_latent_data.py`

## Purpose

Protects the frozen tokenizer-to-prior dataset boundary: normalization must be reversible, atomic
persistence must retain metadata, and source-video leakage must be rejected.

## Components

### `LatentDataTests`
- **Does**: Exercises train-only normalization, save/load roundtrips, and source-disjoint validation.
- **Interacts with**: `latent_data.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Prior experiments | Cached splits remain leakage-safe and denormalizable | Cache validation semantics |
