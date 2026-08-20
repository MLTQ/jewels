# `benchmark_support_renderer.py`

## Purpose

Measures whether the support-complete tiled renderer can replace center-distance KNN during
stage-1 fitting at 10k–72k primitives without changing finite-support results.

## Protocol

- Loads learned geometry from a named fitted checkpoint and fingerprints that exact file.
- Takes seeded random primitive subsets for each requested scale instead of inventing random scale
  distributions.
- Samples query coordinates from the checkpoint's recorded spacetime volume.
- Times synchronized forward + MSE + backward steps, including the tiled index rebuild required
  after geometry changes.
- Reports peak allocated CUDA memory, exact in-support candidate density, and an independent
  pixel audit against the all-center support oracle.
- Records the logical CUDA device name and UUID to avoid ambiguity from CUDA device reordering.

## Gates

The v1 gate requires every requested arm to finish, tiled-vs-oracle maximum pixel error below
`2e-5`, and tiled training-step time no more than 2× KNN at every shared primitive scale. A failed
throughput gate is evidence about this pure-PyTorch implementation and fitted-geometry distribution,
not a universal result about sparse support rendering.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Feasibility report | `support-renderer-benchmark-v1` records, provenance, and explicit gates | Schema or timing scope |
| Renderer | `support` is the audit oracle and `support_tiled` is support-complete | Culling semantics |

## Notes

- Candidate density is a representation property as well as a renderer cost. A world-axis AABB
  narrows the tile hits, then a detached ellipsoid test removes all remaining geometric false
  positives before the autograd gather. The reported count is therefore the set of finite-support
  contributions that a correct renderer still has to process.
- The benchmark catches allocation/runtime failures and records them rather than silently dropping
  an unsuccessful arm.
- The selected v1 run uses an 8,192-point tile query and 1.55 level spacing. Smaller 256-point
  chunks were safe but launch-bound (11.08× KNN at 72k); full-batch indexing converted memory
  headroom on the 24 GB 4090 into throughput. Lower-memory devices should benchmark a smaller chunk.
