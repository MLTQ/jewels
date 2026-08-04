# `test_latent_hierarchy.py`

## Purpose

Protects axis restoration and local block statistics used to choose the scalable prior hierarchy.

## Components

### `LatentHierarchyTests`
- **Does**: Verifies flat raster roundtrips, deterministic 2³ block shape/order, exact recovery of
  block-constant fields by mean pooling, and monotonic PCA explained variance.
- **Interacts with**: `latent_hierarchy.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Hierarchical prior | 2³ blocks flatten to eight latent cells in stable order | Block ordering |
| Research metrics | Block-mean MSE and PCA ratios behave mathematically | Metric definitions |
