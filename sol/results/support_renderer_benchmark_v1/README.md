# Support-complete renderer benchmark v1

## Result

**Label: observed implementation and scaling evidence.** On the pinned RTX 4090, the selected
support-complete tiled renderer matches the all-center five-sigma oracle and completes a synchronized
forward + MSE + backward step within the predeclared 2× KNN threshold at 10k, 45k, and 72k fitted
primitives.

| Primitives | KNN step | Tiled step | Ratio | Exact supports/query | Tiled peak allocation |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 7.56 ms | 14.76 ms | 1.953× | 13.3 | 1.14 GB |
| 45,000 | 20.42 ms | 40.13 ms | 1.965× | 62.8 | 5.01 GB |
| 72,000 | 30.44 ms | 60.55 ms | 1.990× | 100.1 | 8.01 GB |

Maximum tiled-versus-oracle pixel error across scale audits was `1.19e-7`. Local regression tests
also compare every parameter gradient, audit the exact five-sigma boundary, and retain the elongated
time-tilted counterexample and fatal capacity overflow behavior. The 72k run fits on the target
24 GB device with substantial memory headroom.

## Frozen protocol and provenance

- Source checkpoint: a learned 72k-primitive Basketball UCF field, SHA-256
  `0a91ddc63e2e204c3df4396b0b85c0cedf5edac4ad6f202667056f2922e9669e`.
- Geometry subsets: seeded random 10k/45k/72k subsets; seed 20260819.
- Queries: 8,192 sampled volume coordinates; KNN=64; five-sigma finite support.
- Timing: four warmups and 20 synchronized CUDA forward/loss/backward repeats; medians reported.
- Selected tiled settings: base resolution 32, geometric level spacing 1.55, query chunk 8,192.
- Device: NVIDIA GeForce RTX 4090, UUID `21d45575-7ece-a97c-35a0-294f7bce9c39`.

The machine exposes this physical GPU as PyTorch logical device 0 when pinned by UUID. This avoids
the CUDA-ordering ambiguity discovered in earlier runs.

## What changed during the bounded implementation search

The first support-complete version used a longest-axis sphere. It was correct but 12.71× KNN at
72k: 2,228 conservative candidates/query, of which only 4.5% were truly inside the anisotropic
support. That is one implementation result, not a law about spatial indexing.

Adding the exact rotated-ellipsoid AABB reduced the conservative set to 448/query. Raising the query
chunk from 256 to 8,192 removed launch overhead and reduced the 72k ratio from 11.08× to 2.67× at a
7.41 GB peak. Applying the detached true-ellipsoid test before the autograd gather reduced the final
set to 100/query. A bounded 1.2–3.0 level-spacing sweep on the frozen 10k case selected 1.55; the
reported three-scale curve was then rerun independently with 20 repeats.

## Scope

This closes the renderer throughput blocker on one GPU and one learned geometry distribution. It
does not establish multi-device portability or promptable generation. Smaller-memory GPUs need a
separately benchmarked query chunk. The full machine-readable result is in `report.json`.
