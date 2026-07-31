# render.py

## Purpose
Evaluates a primitive field at arbitrary query points: additive anisotropic Gaussian splats
over a learned constant background.

## Components

### `cull_knn(points, mu, k)`
- **Does**: k nearest primitives per query point by Euclidean center distance -> (M,k)
- **Rationale**: Euclidean, not Mahalanobis, on purpose. Culling needs a safe superset only, and
  computing the true metric against all N is the cost we're trying to avoid. Far primitives have
  astronomically negative logits so dropping them is numerically free.

### `render_points(field, points, knn, background)`
- **Does**: (M,3) points -> (M,3) RGB
- **Interacts with**: `PrimitiveField.gather`, called by `fit/fitter.fit_volume`
- **Rationale**: operates on a flat point list rather than a grid, so the same path serves both
  stochastic-voxel training and full-volume inference. Computes `y = S^-1 R^T d` and takes
  `|y|^2` rather than materializing (M,K,3,3) inverse covariances — same result, far less memory.

### `render_volume(field, grid, chunk)`
- **Does**: chunked full-grid render for inference/eval

## Decisions
- The learned constant `background` is load-bearing: it lets a handful of primitives ignore
  flat regions entirely. It lives outside the field (see fitter.md).
- **[2026-07-31] The soft-Voronoi mode was removed.** This file used to implement additive and
  soft-Voronoi as one model under different normalizations; the A/B (including a steelman round
  with a background pseudo-cell + Lloyd relaxation) went to additive on reconstruction AND
  canonicality on real footage. Numbers and reasoning in PROJECT.md's decision log; the
  with-voronoi tree is archived at `jewels/stprim-final-with-voronoi-20260731.tar.gz`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `fit/fitter.py` | `render_points(...) -> (M,3)`, differentiable wrt all field params | Signature, return shape |

## Notes
- Pure PyTorch, no tile rasterizer. This is the throughput bottleneck and the obvious first
  optimization now that the representation question is settled (GSVC-family CUDA rasterizers
  are the reference point).
