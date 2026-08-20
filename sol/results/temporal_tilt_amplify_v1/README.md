# Temporal-tilt real-footage causal ablation v1

## Question

At matched compute and primitive count, does freely tilted spacetime geometry improve a real video
fit relative to axis-aligned spatial/temporal Gaussians?

## Setup

- Source: first 16 frames of `amplify.mp4`, resized to 80×153.
- Both arms use complete five-sigma support rendering.
- 900 optimizer steps, 7.37 million sampled-voxel evaluations, seed 0.
- Both arms end with exactly 791 primitives and take about 133 seconds on an RTX 2070 Super. A
  later UUID audit corrected the original 4090 attribution: CUDA logical device 1 is the 2070
  Super on this host.
- The control retains anisotropic spatial/temporal scales but has every quaternion projected to
  identity throughout fitting.
- The causal gate was declared before the run: at least 0.5 dB advantage, median free mixed tilt at
  least 0.2, and zero mixed tilt in the projected control.

## Result

| Arm | PSNR | Median anisotropy | Median mixed tilt | p90 mixed tilt |
|---|---:|---:|---:|---:|
| Free spacetime geometry | 36.28 dB | 2.22 | 0.372 | 0.940 |
| Axis-aligned control | 34.70 dB | 2.08 | 0.000 | 0.000 |

Free geometry wins by 1.57 dB with the same primitive count, optimizer opportunity, support-safe
renderer, and near-identical wall time. All predeclared causal-gate checks pass.

## Interpretation

Under the project evidence policy, this is a single-source causal signal—not a general law. It is
nonetheless stronger than a post-hoc checkpoint edit: the control was trained end-to-end under its
constraint and had equal opportunity to compensate with centers, scales, colors, and opacity.

Together with the synthetic replication, it supports the claim that mixed space/time orientation
earns reconstruction quality per primitive. Decision-grade evidence still requires at least three
seeds and three motion-diverse clips, including non-fixed-camera footage, with dispersion reported.

The machine-readable measurements are in `report.json`; checkpoints remain in the isolated GPU
workspace at `/home/m/jewels-codex-support/runs/temporal_tilt_amplify_v1`.
