# `axial_prior.py`

## Purpose

Defines a scalable conditional rectified-flow velocity field for the visually validated 16³ block
hierarchy.

## Components

### `AxialConditionalBlock`
- **Does**: Applies attention along one `(u,v,t)` axis and a conditioned pointwise MLP using
  adaLN-Zero gates.
- **Rationale**: A 16-cell line is cheap on the 2070S; rotating axes gives global communication after
  three blocks without allocating 4,096² attention scores.

### `AxialFlowPrior`
- **Does**: Projects 96-D PCA codes, adds separable learned axis positions, cycles axial blocks, and
  predicts same-shape flow velocity from time and nullable CLIP conditions.
- **Interacts with**: Generic flow objectives/sampler in `latent_prior.py` and clamped inpainting.

### Optional mask conditioning
- **Does**: Adds a learned clean/dirty embedding to every coarse cell during repair-aware training.
- **Rationale**: The sampler's clamp mask must be visible to the velocity field; otherwise cells near
  the target end of a flow path do not reveal whether they require velocity or are fixed context.
- **Compatibility**: Older checkpoints omit the feature. For mask-aware models, a missing runtime
  mask means every cell is dirty, preserving full-generation semantics.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Axial trainer | Forward signature matches `RasterFlowPrior` | Condition/drop arguments |
| Flow sampler/inpainting | Output shape exactly matches `(B,4096,96)` input | Velocity shape |
| Masked repair | Explicit dirty mask is embedded per cell | `edit_mask` semantics |
| Checkpoints | Grid shape and every model dimension are serialized | Constructor schema |

## Notes

- Depth need not be a multiple of three, but complete u/v/t sweeps are preferred.
