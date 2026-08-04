# `inpaint.py`

## Purpose

Implements the invariant the editor depends on: generative repair may change dirty latent cells, but
must preserve every clean cell exactly throughout sampling. The condition slot is compatible with a
future text embedding augmented by protected moved-jewel features.

## Components

### `masked_flow_inpaint`
- **Does**: Initializes dirty cells from noise, Euler-integrates a conditional velocity field, and
  reclamps clean cells after every step.
- **Interacts with**: Raster latents from `OccupancyAwareEncoder`; dirty masks from `EditPlan`.
- **Rationale**: Sampling in a fixed raster latent space makes locality and constraint clamping
  explicit, unlike resampling a 45k-element unordered set.

### `VelocityModel`
- **Does**: Documents the callable interface `(latents, time, condition) -> velocity`.

### Mask-aware dispatch
- **Does**: Passes the expanded dirty mask to models declaring `mask_conditioning`; legacy velocity
  callables retain the three-argument interface.
- **Rationale**: The velocity field must distinguish fixed context from cells it is responsible for
  transporting.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Future latent prior | Same-shape velocity output, nullable text, and optional edit-mask support | Callable signature |
| Editor | Clean cells are bit-identical in the returned tensor | Clamp policy |
| Tests | Dirty cells respond to sampling while clean cells never drift | Integration semantics |

## Notes

- This is rectified-flow-style clamping, not the full RePaint jump schedule. Add resampling jumps only
  if measured boundary blending requires them.
- Protected moved jewels must be encoded into the model condition during training; this low-level
  sampler intentionally does not prescribe that encoder.
