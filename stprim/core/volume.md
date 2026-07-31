# volume.py

## Purpose
Turns a (T,H,W) video shape into normalized (u,v,t) coordinates, and owns the one genuinely
arbitrary choice in the representation: the exchange rate between pixels and frames.

## Components

### `make_grid(shape, t_scale, ...)`
- **Does**: (T,H,W) -> (T*H*W, 3) coords in [-1,1]^3, C-order
- **Rationale**: ordering deliberately matches `video.reshape(-1,3)` for a (T,H,W,3) tensor so
  target lookup is a plain index, no permutation.

### `sample_indices(n_total, n_sample, ...)`
- **Does**: uniform random voxel indices for stochastic fitting
- **Rationale**: fitting against random voxels rather than whole frames keeps VRAM flat
  regardless of clip length — the practical payoff of the volume framing.

## Decisions
- Each axis normalized to [-1,1] independently, so non-square video is implicitly aspect-warped.
  Accepted: per-primitive anisotropy absorbs it. Revisit if it interacts badly with kNN culling.
- `t_scale` multiplies the t axis *on top of* normalization, so it means "how many pixels is a
  frame worth" in a relative sense, not absolute.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `fit/fitter.py` | grid row order matches `video.reshape(-1,3)` | Meshgrid indexing order |

## Notes
- A mis-set `t_scale` is absorbed by per-primitive anisotropy during optimization but NOT by kNN
  culling, which is isotropic. First knob to check on high-motion footage.
