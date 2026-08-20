# render.py

## Purpose
Evaluates a primitive field at arbitrary query points: additive anisotropic Gaussian splats
over a learned constant background.

## Components

### `cull_knn(points, mu, k, chunk)`
- **Does**: k nearest primitives per query point by Euclidean center distance -> (M,k)
- **Status**: historical throughput baseline only. Center-distance KNN is not a safe superset for
  anisotropic splats: a long, tilted primitive can contribute strongly at a query while more than
  `k` narrow primitives have closer centers. It remains the default temporarily so old fits and
  recovery files keep their meaning, but evidence-bearing runs must state the culling mode.
- Chunked over query points (2026-08-01): bit-identical indices, peak memory chunk×N instead of
  M×N. The unchunked 65536×10000 matrix was the fitter's 13 GB spike; now ~2-3 GB total, so
  fits can share a GPU. Since 2026-08-04, the effective chunk is also capped at 100M
  point/primitive distance pairs. It automatically shrinks the chunk as primitive count grows,
  preventing the 2.5 GB
  contiguous allocation and subsequent large-kernel launch failure observed during two 45k UCF
  transfer attempts on the 8 GB 2070S. After a device restart, an 8,192×45,000 stress test used
  394 MB peak PyTorch allocation and a full 8,000-step/45k UCF fit completed with ~1.36 GB reported
  GPU use. A subsequent benchmark over 65,536 queries and 45k centers measured 0.513/0.408/0.387/
  0.387 seconds at 50M/75M/100M/125M caps. The 100M result used 492 MB peak and returned identical
  neighbors; 64 consecutive repetitions completed in 25.0 seconds with 404 MB peak reserved. The
  100M cap is therefore the selected throughput/stability knee. The original 50M recovery run took
  57 minutes, so corpus throughput still needs a full-fit remeasurement. An end-to-end call through
  `cull_knn` on 65,536 queries and 45k centers measured 0.522 s / 423 MiB at the old 50M chunk and
  0.390 s / 837 MiB at 100M, with identical `(65536,64)` indices. The additional workspace is safe
  on the 8 GB card and buys about 25% lower culling latency.

### `cull_support_sphere(points, mu, max_scale, support_sigma, capacity)`
- **Does**: returns every primitive that can fall within a declared finite Mahalanobis support.
- **Proof**: `||x-mu|| / max(scale)` is a lower bound on Mahalanobis radius because rotation
  preserves Euclidean norm and no principal scale exceeds `max(scale)`. Selecting all conservative
  spheres within `support_sigma`, then applying the true metric, cannot omit an in-support splat.
- **Failure behavior**: probes `capacity + 1` candidates and raises `SupportOverflowError` if the
  budget cannot be proven complete. This converts a silent scientific error into an actionable
  capacity setting.
- **Approximation**: Gaussian mass beyond the declared support is set to zero. At the default
  five-sigma boundary an individual boundary weight is `exp(-12.5)`, about 3.7e-6.

### `render_points(field, points, knn, cull_mode, support_*, background)`
- **Does**: (M,3) points -> (M,3) RGB
- **Interacts with**: `PrimitiveField.gather`, called by `fit/fitter.fit_volume`
- **Rationale**: operates on a flat point list rather than a grid, so the same path serves both
  stochastic-voxel training and full-volume inference. Computes `y = S^-1 R^T d` and takes
  `|y|^2` rather than materializing (M,K,3,3) inverse covariances — same result, far less memory.
- `cull_mode="knn"` is the legacy approximation, `"support"` is finite-support correct with an
  explicit candidate budget, and `"exact"` renders all primitives for tiny reference audits.
- Support rendering chunks query points before gathering parameters, bounding the large
  `(points, candidates, parameters)` workspace while preserving gradients.

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
- **[2026-08-19] Support-correct audit path added.** The previous documentation incorrectly called
  Euclidean center-KNN a safe superset. A constructed elongated-splat counterexample disproves
  that claim. Five-sigma conservative-sphere culling is now the correctness path; KNN is retained
  only to make matched A/B tests and old checkpoints reproducible.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `fit/fitter.py` | `render_points(...) -> (M,3)`, differentiable wrt all field params | Signature, return shape |

## Notes
- Pure PyTorch, no tile rasterizer. This is the throughput bottleneck and the obvious first
  optimization now that the representation question is settled (GSVC-family CUDA rasterizers
  are the reference point).
