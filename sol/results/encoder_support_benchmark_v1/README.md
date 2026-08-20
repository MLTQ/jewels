# Encoder support-renderer benchmark

A mature 73,728-splat encoder output was rendered for a synchronized forward + MSE + backward
step on the pinned RTX 4090. The support-complete five-sigma path is compared with the all-center
exact implementation used as a correctness oracle.

| Renderer | Median step | Peak allocation |
|---|---:|---:|
| All-center exact | 856.05 ms | 5.96 GiB |
| Support-complete tiled | 34.75 ms | 0.78 GiB |

The tiled path is 24.64× faster and differs from the infinite-tail oracle by at most `1.19e-5` in
pixel value; that tiny difference is the declared five-sigma truncation, not missing in-support
contributors. `benchmark.png` plots runtime and memory; `report.json` records all synchronized
repeats and settings.
