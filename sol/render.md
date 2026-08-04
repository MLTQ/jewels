# `render.py`

## Purpose

Provides a correctness reference for jewel rendering and replaces unsafe Euclidean center-kNN
culling with an explicit, auditable finite-support rule. This is research code: correctness and a
declared approximation boundary come before rasterizer speed.

## Components

### `covariance_terms`
- **Does**: Converts canonical log-covariance feature dimensions into covariance and precision.
- **Interacts with**: The 22-value layout in `stprim/prior/featurize.py`.
- **Rationale**: Eigenvalues clamp to the production scale bounds before exponentiation, preventing
  early autoencoder predictions from creating infinite covariances.
- **Scaling**: Eigensolves in bounded chunks because cuSOLVER requests an 11+ GB workspace for a
  single 45k×3×3 batch on the 8 GB 2070S even though the input/output tensors are small.
- **Boundary**: Empty decoded sets return empty covariance/precision tensors so untrained count
  models remain evaluable.

### `render_exact`
- **Does**: Evaluates every Gaussian at every query point in bounded-memory chunks.
- **Rationale**: Serves as the oracle for culling and future CUDA-rasterizer tests.

### `render_truncated`
- **Does**: Truncates at a declared Mahalanobis radius after a conservative covariance-derived AABB
  prefilter.
- **Rationale**: A point inside the ellipsoid cannot fall outside the AABB, even for a distant,
  highly elongated, tilted jewel.

### `render_euclidean_knn`
- **Does**: Reproduces the existing center-kNN behavior for regression demonstrations.
- **Rationale**: This is a negative-control baseline, not the intended renderer.

### `audit_truncation` / `RenderAudit`
- **Does**: Reports finite-support error against the all-jewel reference.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `spike.py` | Exact and approximate renderers accept `(N,22)` features | Feature layout or signatures |
| `tests/test_render.py` | Truncated renderer retains broad elongated jewels | Culling semantics |
| Dense tokenizer | Covariance conversion remains differentiable with bounded solver workspace | Chunk semantics |

## Notes

- Gaussian support is mathematically infinite. `support_sigma` makes truncation explicit rather than
  pretending a fixed count of nearest centers is exact.
- This implementation still forms point-by-primitive blocks. A production rasterizer should index
  AABBs spatially while remaining bit-comparable to this reference.
