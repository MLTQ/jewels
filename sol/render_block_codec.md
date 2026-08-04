# `render_block_codec.py`

## Purpose

Provides the visual gate for the fixed 16³/96-D hierarchy before any generative prior is trained.

## Components

### `main`
- **Does**: Reconstructs fine latents from aligned PCA codes, sparse-decodes jewels, and writes
  source-balanced fitted-target/hierarchical-roundtrip GIFs plus a manifest.
- **Interacts with**: `block_codec.py`, the fine/coarse caches, and production render helpers.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Hierarchical prior decision | Panel order is fitted target / PCA hierarchy roundtrip | Visual semantics |
| Research record | Manifest carries source, frame picks, decoded count, and artifact | Manifest schema |
