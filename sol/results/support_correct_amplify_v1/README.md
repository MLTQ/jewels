# Support-correct real-footage scaling v1

## Question

Does the corrected fitter continue to improve with compute and learn genuinely anisotropic,
time-oriented primitives on real footage?

## Setup

- Source: first 16 frames of `amplify.mp4`, resized to 80×153.
- Same seed, sampled voxels, initialization, adaptation, and primitive budget in both arms.
- Step budgets: 100, 300, and 900; final primitive counts: 500, 583, and 791.
- Legacy arm: 64 nearest Euclidean centers.
- Corrected arm: complete five-sigma candidates with capacity 1,024 and fatal overflow.
- Every checkpoint is evaluated using the corrected renderer.
- Hardware: RTX 4090.

## Result

| Steps | Legacy support-audit PSNR | Corrected PSNR | Median anisotropy | Corrected time |
|---:|---:|---:|---:|---:|
| 100 | 24.67 dB | 24.67 dB | 1.02 | 11.2 s |
| 300 | 30.04 dB | 30.04 dB | 1.32 | 36.4 s |
| 900 | 36.00 dB | 36.28 dB | 2.22 | 133.4 s |

The corrected arm gains 11.61 dB from 100 to 900 steps and passes the predeclared median-anisotropy
gate. At 900 steps, p90 anisotropy is 5.67 and p90 longest-axis temporal alignment is 0.997. This is
direct evidence that the fitter increasingly uses elongated spacetime elements rather than merely
placing isotropic video samples.

Unlike the synthetic counterexample, KNN does not catastrophically bias aggregate PSNR on this
small real clip: corrected training is only 0.28 dB better at 900 steps. It does still create local
disagreement, with a 0.093 maximum pixel difference between the KNN training renderer and the
support-safe audit.

## Interpretation

The representation now has two positive signals needed for the compute-feasibility case:

1. full-volume quality improves monotonically with optimization compute; and
2. learned geometry becomes substantially more anisotropic and time-oriented as compute rises.

This does not yet prove the geometry causes the quality gain. The next causal experiment must fit
the same clip with temporal tilt projected out, at the same primitive and compute budget. The
corrected renderer is also 8.9× slower than KNN at 900 steps (133.4 s versus 15.1 s), making a
support-safe tiled CUDA rasterizer the main scale-up dependency.

The machine-readable measurements are in `report.json`; checkpoints remain in the isolated GPU
workspace at `/home/m/jewels-codex-support/runs/support_correct_amplify_v1`.
