# Temporal-tilt causal ablation v1

## Question

At matched compute and primitive count, does freely rotated spacetime geometry reconstruct motion
better than axis-aligned geometry that cannot represent a sheared trajectory with one Gaussian?

## Setup

- Source: deterministic translating synthetic tube, 16 frames at 64×64.
- Both arms use complete five-sigma support rendering.
- `free` learns arbitrary scales and rotations.
- `axis_aligned` retains independent spatial and temporal scales but projects every quaternion to
  identity throughout fitting and after densification.
- Same seed, voxel samples, optimizer, adaptation, and primitive counts.
- Predeclared gate: at least 0.5 dB free-geometry advantage, median free mixed tilt at least 0.2,
  and numerically zero mixed tilt in the projected control.
- Hardware: RTX 4090.

## Result

| Steps | Primitives | Free PSNR | Axis-aligned PSNR | Free advantage | Free median mixed tilt |
|---:|---:|---:|---:|---:|---:|
| 300 | 349 | 48.26 dB | 47.29 dB | 0.97 dB | 0.519 |
| 900 | 473 | 58.01 dB | 56.96 dB | 1.05 dB | 0.508 |

All causal-gate checks pass. The control's median and p90 mixed tilt are exactly zero, while the
free arm's p90 mixed tilt is 0.95 at 900 steps. Both arms receive equal optimizer compute and end
with exactly the same primitive count, so the measured advantage is attributable to access to
mixed space/time orientation, not extra elements or a different renderer.

## Interpretation

This is the first causal evidence for the project's defining representation claim. Arbitrarily
oriented spacetime Gaussians buy roughly 1 dB on a literal motion tube, and that advantage persists
as compute triples from 300 to 900 steps rather than being optimized away.

It is still a synthetic falsification test. The next pitch-critical gate is the same ablation on
several held-out real motion clips and seeds, reporting confidence intervals and quality per byte.
The machine-readable measurements are in `report.json`; checkpoints remain in the isolated GPU
workspace at `/home/m/jewels-codex-support/runs/temporal_tilt_synthetic_v1`.
