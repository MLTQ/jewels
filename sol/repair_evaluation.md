# `repair_evaluation.py`

## Purpose

Defines the reproducible gate for editor-specific training: can the axial prior reconstruct noisy
cuboid holes from exact surrounding latent context?

## Components

### `sample_cuboid_masks`
- **Does**: Samples filled coarse-grid cuboids with bounded random extents and locations.
- **Rationale**: A translated parallelepiped plus its sweep becomes a conservative cuboid-like dirty
  region after rasterization.

### `evaluate_masked_repair`
- **Does**: Runs clamped flow repair on fixed held-out masks and reports dirty-region latent MSE,
  normalized zero-fill MSE, clean-cell error, and mask fraction.
- **Rationale**: Full-generation metrics do not measure the editor state distribution.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Masked trainer | Cuboids flatten in canonical `(u,v,t)` order | Mask layout |
| Experiment comparison | Same seed, examples, extents, and Euler steps | Evaluation protocol |
| Editor invariant | Clean error is exactly zero | Sampler clamping |
