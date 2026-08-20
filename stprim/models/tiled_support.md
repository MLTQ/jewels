# `tiled_support.py`

## Purpose

Builds and queries a support-complete multilevel spatial index for anisotropic spacetime splats.
It replaces all-query/all-center distance matrices without weakening the five-sigma correctness
contract.

## Components

### `SupportOverflowError`
- **Does**: signals that a query's conservative support-bound candidates exceed the declared budget.

### `TileLevel` / `SupportTileIndex`
- **Does**: immutable detached geometry index: one sorted center bin per primitive at one support
  level, plus tight world-axis and detached ellipsoid geometry (or a spherical fallback).

### `build_support_tile_index(mu, max_scale, half_extent, support_sigma, base_resolution, level_scale)`
- **Does**: assigns every primitive to the smallest geometric cell width not smaller than the
  largest component of its support AABB (or its conservative sphere without an AABB).
- **Proof**: an ellipsoid is contained by half-extent
  `sigma * sqrt(sum_j R[axis,j]^2 scale[j]^2)` on each world axis. If the cell width is at least
  the largest half-extent, an in-support query and center differ by at most one cell on every axis.
  Querying the 27 neighboring cells and filtering by the AABB therefore cannot omit the ellipsoid.
- **Rationale**: each primitive is stored exactly once. Broad primitives move to coarse levels
  instead of being duplicated into thousands of fine tiles.

### `query_support_pairs(index, points, capacity)`
- **Does**: uses packed signed xyz keys and `searchsorted` to retrieve neighboring bins, applies the
  conservative AABB (or sphere fallback), then applies the detached true-ellipsoid test before
  enforcing capacity and returning ragged pairs.
- **Rationale**: the exact detached filter prevents loose bounding-box hits from entering the much
  larger autograd parameter gather. The renderer recomputes the same metric from live parameters,
  in the same operation order, preserving boundary semantics and gradients for every selected
  in-support contribution.
- **Rationale**: ragged pairs avoid padding every query to the worst-case candidate count.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `models/render.py` | Returned pairs contain every primitive whose true Mahalanobis radius can be inside support | Bound formula, binning proof, key packing, pair semantics |
| Renderer tests | Overflow is loud and the time-tilted counterexample survives | Capacity or support semantics |

## Notes

- Coordinates are packed as three signed 21-bit integers. Values outside that very large range fail
  rather than collide.
- Index construction and candidate selection use detached geometry, like KNN/top-k selection; the
  selected render contributions remain differentiable with respect to field parameters.
- The default 1.55 geometric level spacing was selected by a bounded 1.2–3.0 sweep on the frozen
  10k subset. The final 10k/45k/72k curve was rerun separately with 20 timing samples.
