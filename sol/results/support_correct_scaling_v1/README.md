# Support-correct scaling experiment v1

## Question

Does the stage-1 spacetime-splat fitter improve with compute when its renderer is prevented from
silently omitting elongated contributors?

## Setup

- Source: deterministic translating synthetic tube, 16 frames at 64×64.
- Same seed, sampled voxels, initialization, adaptation schedule, and primitive budget in both arms.
- Step budgets: 100, 300, and 900; final primitive counts: 300, 349, and 473.
- Legacy arm: 64 nearest Euclidean centers.
- Corrected arm: complete candidate set inside a five-sigma Mahalanobis support, with capacity 512
  and fatal overflow.
- Every checkpoint is evaluated using the corrected renderer.
- Hardware: RTX 4090.

## Result

| Steps | Legacy training PSNR | Legacy support-audit PSNR | Corrected PSNR | Corrected time |
|---:|---:|---:|---:|---:|
| 100 | 32.06 dB | 32.06 dB | 32.06 dB | 7.1 s |
| 300 | 48.27 dB | 48.26 dB | 48.26 dB | 22.8 s |
| 900 | 53.39 dB | 42.51 dB | 58.01 dB | 82.8 s |

At the largest budget, center-KNN creates an 10.88 dB self-evaluation illusion: its training
renderer reports 53.39 dB, but complete support evaluation reports 42.51 dB and finds a maximum
pixel difference of 0.340. Support-correct training reaches 58.01 dB with the same 473 primitives
and optimizer budget. Its corrected PSNR rises monotonically by 25.94 dB across the compute curve.

## Interpretation

This is strong evidence that the corrected representation/fitter responds to compute and that the
old renderer can corrupt late, densified fits. It is not yet the pitch gate for promptable
text-to-video:

- The synthetic reconstruction and compute-slope gates pass.
- Median anisotropy is 1.30, below the predeclared 1.5 structural gate, although p90 anisotropy is
  1.74 and p90 principal-axis temporal alignment is 0.80.
- The corrected path is about 5.4× slower at 900 steps (82.8 s versus 15.4 s), so a tile/bin CUDA
  implementation is required before large corpus fitting.
- A real-footage repeat and a no-temporal-tilt ablation remain required before attributing the gain
  specifically to time-distorted splats.

The machine-readable measurements are in `report.json`; checkpoints remain in the isolated GPU
workspace at `/home/m/jewels-codex-support/runs/support_correct_scaling_v1`.
