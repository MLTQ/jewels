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
- `cull_mode="knn"` is the legacy approximation, `"support"` is the all-center finite-support
  reference, `"support_tiled"` uses the support-complete multilevel index in `tiled_support.py`,
  and `"exact"` renders all primitives for tiny reference audits.
- Support rendering chunks query points before gathering parameters, bounding the large
  `(points, candidates, parameters)` workspace while preserving gradients.
- Tiled support uses the exact world-axis bounding box of each rotated support ellipsoid before
  gathering full primitive parameters, then reduces ragged pairs with `index_add`. The true
  Mahalanobis test is applied once with detached index geometry before the autograd gather and again
  with live parameters in this file. The AABB is much tighter than a longest-axis sphere for
  elongated spacetime splats, and the detached exact filter removes its remaining false positives;
  neither step weakens completeness.

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
- **[2026-08-19] Multilevel tiled path added.** Radius-matched center bins preserve support
  completeness while storing each primitive once. The all-center support mode remains the oracle
  until tiled output, gradients, memory, and end-to-end throughput pass the scale gate.
- **[2026-08-19] Exact pre-gather filtering.** A spherical first version passed correctness but was
  12.71× KNN at 72k because 95.5% of its candidates were false positives for elongated ellipsoids.
  A support AABB plus detached exact test reduces the measured 72k set from 2,228 to 100 candidates
  per voxel and passes the separate 2× training-step gate. This comparison is implementation-specific,
  not a negative conclusion about spherical indexing in other representations.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `fit/fitter.py` | `render_points(...) -> (M,3)`, differentiable wrt all field params | Signature, return shape |
| `models/tiled_support.py` | Ragged candidates are conservatively complete; this file applies true q and sums contributions | Pair semantics |

## Notes
- The tiled path is pure PyTorch. Passing correctness does not imply it reaches the 2× KNN
  throughput gate; benchmark before selecting it for corpus generation.
