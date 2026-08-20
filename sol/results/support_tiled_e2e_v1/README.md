# Tiled support end-to-end optimizer check v1

## Result

**Label: observed implementation evidence.** On the same 16-frame PlayingGuitar clip, seed,
500-primitive budget, 819,200 sampled voxel evaluations, and optimizer settings, tiled support and
the all-center support oracle reach effectively identical results after 100 optimizer steps:

| Renderer | Support-evaluated PSNR | Fit time | Gap from support oracle |
|---|---:|---:|---:|
| All-center support | 19.416349 dB | 3.729 s | 0 |
| Tiled support | 19.416363 dB | 0.694 s | max `3.58e-7` RGB |

The fields retain matching structure statistics; for example median anisotropy is 1.06318 in both
arms. The 5.37× speedup is relative to the correctness oracle, not KNN. The separate 10k–72k
training-step benchmark is the KNN throughput gate.

## Scope

This positive control verifies that the tiled selection path remains differentiable and behaves
the same inside optimization, beyond unit-level gradient comparisons. It is one clip and seed, so
it is not a quality or generalization experiment. The machine-readable record is in `report.json`.
