# Frozen geometry and perceptual appearance v1 — registered protocol

## Question

Can the best residual irregular encoder cross exact `20 dB` and improve LPIPS by training only its
appearance branch with contiguous full-frame supervision, while centers, covariance, and opacity
remain bitwise identical to the already-passing source?

## Source and invariants

- Source: `appearance_contract_v1/continuation_2070/control_total1200/encoder.pt`, exact seed-0
  `19.66478 dB / 0.73288` LPIPS, sampled `19.90453 dB`, occupancy `0.984577`, active fraction
  `0.628149`, and mixed spacetime tilt `0.521151`.
- Architecture remains `factorized_structural_jewel_encoder_v3`, residual appearance contract,
  `16 x 16 x 8 x 20 = 40,960` proposals.
- `--freeze-geometry` excludes the geometry trunk/head from the optimizer. Before step one, the
  trainer snapshots held-out centers, log-scales, quaternions, and opacity logits; every evaluation
  must report `bitwise_exact=true` and `max_abs_change=0` on all five sources.
- Geometry teacher, sparsity, density, orientation, local, and responsibility weights are all zero.
  The existing geometry therefore cannot be repaired, regularized, or accidentally de-gridded by
  any registered arm.

## Read-only calibration

`calibration_seed0.json` uses the source checkpoint, the first declared training clip, seed 0,
1,024 sampled render points, and four contiguous `48 x 72` frames. It constructs no optimizer and
takes no step.

| Component | Raw loss | Appearance gradient L2 |
|---|---:|---:|
| sampled render MSE | 0.015491 | 0.019135 |
| grid multiscale RGB | 0.077011 | 0.173811 |
| grid spatial edge | 0.078619 | 0.044027 |
| grid temporal edge | 0.031445 | 0.065871 |
| grid spatiotemporal structure | 0.179658 | 0.854514 |
| sampled rendered-range excess | 0.00000350 | 0.001061 |
| residual RGB energy | 0.047487 | 1.000974 |
| residual Jacobian energy | 0.012337 | 0.221922 |

Sampled rendered RGB is outside `[0,1]` in `0.8138%` of elements (`0.7487%` below zero). Geometry
has zero trainable parameters; appearance has 64,223. Registered weights below keep the expected
full-frame gradient near half the sampled-render gradient after one-in-four frequency correction;
range/residual regularization remains a small stabilizer rather than reinstating hard bounds.

## Matched seed-0 arms

Every arm starts from the same source and receives 600 steps, seed 0, 120-video manifest, AdamW peak
LR `1e-4`, warmup 100, 1,024 sampled points, support-tiled capacity 8,192, identical source
selection, and five held-out validation sources.

| Arm | Grid total | Internal grid weights RGB/spatial/temporal/structure/range | Sampled range | Residual RGB/Jacobian |
|---|---:|---|---:|---|
| frozen-render | 0 | — | 0 | 0 / 0 |
| frozen-perceptual | 0.05 | 1 / 0.5 / 0.25 / 0.01 / 1 | 0 | 0 / 0 |
| frozen-stabilized | 0.05 | 1 / 0.5 / 0.25 / 0.01 / 1 | 1.0 | 0.0005 / 0.001 |

Grid updates use four contiguous `48 x 72` frames every four steps and multiply the scheduled update
by four, so the declared total weight describes its expected per-step contribution. The control
does not render the extra grid; frame selection is deterministic and consumes no training RNG.

## Decision rules

All finite, geometry-exact arms within `0.50 dB` sampled PSNR of frozen-render are exact-audited
together. Structure must remain at the source values and retain occupancy `<=0.985`, active fraction
`<=0.70`, and mixed tilt `>=0.25`.

The perceptual intervention passes if it improves both exact macro PSNR and LPIPS over the source
and frozen-render, with joint PSNR/LPIPS improvement in at least four of five styles. The stabilized
arm passes causally if it reduces rendered out-of-range fraction, residual energy, and target-
relative temporal-change overshoot without losing more than `0.05 dB` PSNR or `0.002` LPIPS against
frozen-perceptual. Visual sheets must show fewer isolated dark speckles and no loss of recognizable
boundaries.

Absolute appearance promotion requires exact macro PSNR `>=20 dB`; LPIPS `<=0.40` remains the final
representation gate, not an expected result of one tranche. If one arm crosses 20 dB, improves both
metrics over frozen-render, and passes visual/stability review, repeat that arm from the same source
with continuation seeds 1 and 2. This tests appearance-optimization robustness, not independent
upstream encoders. A failed arm scopes only these weights, source, and 600-step budget.

## Compute ownership

Calibration ran on the physical RTX 4090 while it was idle. Registered training uses the physical
RTX 2070 Super after cleanly stopping and runtime-masking `scroller-llm.service`; the service must be
unmasked and restarted after the experiment. GPU runtime is operational data, not model evidence.
