# `__init__.py`

## Purpose

Defines the small public surface of the research spike. Importing `sol` exposes geometry,
structured-grid, and inpainting primitives without importing the runnable demonstration.

## Components

### Public exports
- **Does**: Re-exports `GridSpec`, `OccupancyGrid`, `Parallelepiped`, `translate_selected`, and
  `masked_flow_inpaint`, `RasterFlowPrior`, and `StructuredJewelAutoencoder`.
- **Interacts with**: `geometry.py`, `token_grid.py`, `autoencoder.py`, `inpaint.py`, and
  `latent_prior.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Tests and future training CLIs | Stable top-level names for core spike concepts | Removing or renaming an export |
