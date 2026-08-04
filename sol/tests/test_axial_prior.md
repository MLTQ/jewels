# `test_axial_prior.py`

## Purpose

Protects the scalable hierarchy prior's compatibility with existing flow training and sampling.

## Components

### `AxialPriorTests`
- **Does**: Exercises a complete u/v/t sweep, nullable conditioning, backpropagation, and generic
  Euler flow sampling while preserving latent shape, architecture-aware evaluator restoration, and
  explicit edit-mask validation.
- **Interacts with**: `axial_prior.py` and `latent_prior.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Trainer and editor | Axial prior is a drop-in velocity model | Forward signature |
| Hierarchical cache | Flattened cell count equals the product of grid axes | Grid validation |
| Masked repair trainer | Clean/dirty mask has one value per cell and batch item | Mask validation |
