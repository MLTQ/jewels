# `block_codec.py`

## Purpose

Defines the first reversible hierarchy above the visually successful grid-32 tokenizer: a fixed PCA
code for each non-overlapping 2³ latent block.

## Components

### `BlockPCACodec`
- **Does**: Encodes eight fine cells into one coarse code and reconstructs the original canonical
  raster ordering.
- **Interacts with**: `latent_hierarchy.block_vectors` and tokenizer latent normalization.
- **Rationale**: A fixed linear codec makes the hierarchy measurable before a learned block decoder
  can hide capacity or overfit absolute cell identities.

### `fit_block_pca`
- **Does**: Fits the mean and top covariance eigenvectors from training-source blocks only.
- **Rationale**: Deterministic evenly spaced block subsampling bounds memory without crossing the
  held-out source boundary.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Coarse cache | Codes are `(sample,coarse_u*coarse_v*coarse_t,code_dim)` | Code layout |
| Fine tokenizer | `decode(encode(x))` restores `(sample,32³,latent_dim)` order | Inverse block permutation |
| Axial prior | Coarse shape and explained variance travel with the codec | State schema |
