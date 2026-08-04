# `latent_hierarchy.py`

## Purpose

Measures whether dense raster latents support a coarse-to-local generative hierarchy before a prior
architecture is chosen.

## Components

### `reshape_latents`
- **Does**: Restores `(u,v,t)` axes from flat canonical cell order.
- **Interacts with**: `GridSpec.cell_index` ordering and cached tokenizer latents.

### `axis_neighbor_correlations`
- **Does**: Measures adjacent-cell Pearson correlation independently along image width, height, and
  time.

### `block_vectors`
- **Does**: Converts non-overlapping cubic regions into rows with stable local-cell ordering.
- **Rationale**: The same layout can later feed a shared block encoder/decoder and local residual
  flow.

### `hierarchy_report`
- **Does**: Reports block-mean reconstruction error and PCA explained variance for candidate coarse
  code widths.
- **Interacts with**: `analyze_latent_hierarchy.py` and the future hierarchical prior.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Hierarchy analysis | Flat order reshapes exactly to checkpoint grid metadata | Axis ordering |
| Block prior | Block vectors use `(local_u,local_v,local_t,dimension)` order | Permutation/layout |
| Research record | MSE is measured on train-normalized latents | Normalization semantics |
