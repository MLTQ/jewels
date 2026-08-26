# `__init__.py`

## Purpose

Defines the small public surface of the research spike. Importing `sol` exposes stable names for
geometry, structured-grid, and inpainting primitives without eagerly importing the Torch training
stack or the runnable demonstration.

## Components

### Public exports
- **Does**: Re-exports `GridSpec`, `OccupancyGrid`, `Parallelepiped`, `translate_selected`, and
  `masked_flow_inpaint`, `RasterFlowPrior`, and `StructuredJewelAutoencoder`.
- **Interacts with**: `geometry.py`, `token_grid.py`, `autoencoder.py`, `inpaint.py`, and
  `latent_prior.py`.
- **Import contract**: Uses module-level lazy attribute resolution. Lightweight tooling can import
  `sol.*` modules on CPU-only environments; requesting a training export still imports its original
  implementation and dependencies.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Tests and future training CLIs | Stable top-level names for core spike concepts | Removing or renaming an export |
