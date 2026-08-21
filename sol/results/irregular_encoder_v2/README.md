# Factorized irregular jewelfield v3 — experiment report

## Outcome

This round resolves the original grid diagnosis and replaces it with a narrower, evidence-backed
problem statement:

- **The encoder can learn a genuinely irregular, sparse, time-tilted field.** All three tested
  proposal capacities pass the preregistered structure screen, and the center-layout image no
  longer shows an XY or XT lattice.
- **More fixed proposals do not improve fidelity.** Moving from 20,480 to 73,728 proposals leaves
  sampled held-out PSNR essentially flat (`17.09–17.29 dB`).
- **The visible blur is partly caused by excessive jewel extent.** The selected field's median
  extent is `0.05158`, 4.01× the fitted-teacher median of `0.01285`. The old scale-invariant spread
  objective could not detect this.
- **Absolute size is controllable, but it exposes an appearance/coverage tradeoff.** A registered
  weight of `0.023` reduces median extent 21.6% while remaining within 0.25 dB of its sampled
  control and preserving every structure limit. In a side-by-side exact audit, LPIPS improves
  2.95%, while PSNR and SSIM decline.
- **This is not yet a promotable text-to-video result.** Exact candidate quality remains about
  `18 dB` PSNR and `0.79–0.82` LPIPS, versus a `23.65 dB` lattice encoder and a `27.70 dB` fitted
  teacher. No text-conditioned generator has yet been trained over the irregular representation.

The feasibility conclusion is therefore asymmetric: irregular time-distorted Gaussian splats are
now demonstrated as a learnable video representation, but “more compute on the current loss” is
not enough evidence for a text-to-video pitch. The next proof must distill local teacher topology
and appearance without restoring a lattice.

## What changed

`factorized_structural_jewel_encoder_v3` separates the geometry trunk/head from a fine/coarse
appearance path. Appearance samples use detached centers, so image gradients cannot pull geometry
back toward a raster. It keeps mobile stratified centers, quaternion/log-scale covariance, direct
mixed-spacetime tilt, opacity-weighted density matching, and explicit sparsity.

The trainer now supports:

- v2 geometry transplant into the factorized architecture;
- ordinary geometry freezing for appearance-only continuation;
- a bounded multiscale image/edge loss;
- absolute mean-log-scale supervision with a declared teacher offset;
- median jewel extent in held-out evaluation records.

The exact auditor now restores both v2 and v3 checkpoints. A discovered audit defect that scored
multiple candidates but drew only the first has been fixed; qualitative and center-layout figures
now contain every supplied candidate.

The `v3_smoke5` and `v3_gridloss_smoke4` directories are integration smokes only: they verify that
the new architecture and full-image objective execute and checkpoint. Their four/five-step values
are not treated as scientific evidence or as failed training results.

## Capacity screen

All arms use seed 0, 600 steps, the same 120 training videos, 36 fitted training teachers, and five
held-out styles. See [PROTOCOL.md](PROTOCOL.md) for the frozen decision rule.

| Slots/cell | Proposals/window | Sampled PSNR | Occupancy | Active | Mixed tilt | Eligible |
|---:|---:|---:|---:|---:|---:|:---:|
| 10 | 20,480 | 17.2566 | 0.97485 | 0.62892 | 0.49541 | yes |
| 20 | 40,960 | **17.2945** | **0.97053** | 0.62620 | **0.51171** | **yes, selected** |
| 36 | 73,728 | 17.0921 | 0.97984 | 0.67217 | 0.48625 | yes |

The count curve is flat, which rejects fixed proposal capacity as the immediate bottleneck for
this configuration. The 20-slot arm was selected by the preregistered highest-PSNR rule.

## Appearance continuation and exact audit

With geometry frozen, the selected arm improves sampled held-out PSNR from `17.2945` at 600 steps
to `18.3234` at 6,000 total steps. Exact quality does not follow far enough:

| Arm | Exact PSNR | LPIPS | SSIM | Occupancy | Active | Mixed tilt |
|---|---:|---:|---:|---:|---:|---:|
| Factorized v3, 6k | 17.9549 | 0.8123 | 0.7076 | 0.97048 | 0.62620 | 0.51178 |
| Lattice reference | 23.8683 | 0.3916 | 0.9267 | 0.99919 | 1.00000 | 0.24814 |
| Fitted teacher | 27.6993 | 0.1720 | 0.9680 | — | — | — |

The structural checks pass and both perceptual checks fail. The 5,400-step frozen continuation
adds only `0.2525 dB` after its first evaluation, so a longer appearance-only run is not the best
next use of compute.

## Absolute-size causal test

See [SIZE_PROTOCOL.md](SIZE_PROTOCOL.md) for the matched control, bracket, and boundary
confirmation registered before each execution.

| Size weight | Sampled PSNR | Median extent | Change vs control | Occupancy | Active | Mixed tilt | Pareto pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | **18.3122** | 0.05093 | — | 0.97109 | 0.63947 | 0.52211 | no: size |
| 0.01 | 18.2901 | 0.04637 | −8.96% | 0.97340 | 0.65549 | 0.52340 | no: size |
| 0.02 | 18.1554 | 0.04158 | −18.36% | 0.97568 | 0.67102 | 0.52369 | no: size |
| **0.023** | **18.1175** | **0.03995** | **−21.57%** | **0.97593** | **0.66966** | **0.52350** | **yes** |
| 0.03 | 18.0087 | 0.03622 | −28.90% | 0.97626 | 0.67478 | 0.52589 | no: PSNR |
| 0.05 | 17.7263 | 0.02636 | −48.25% | 0.97857 | 0.67446 | 0.52218 | no: PSNR |

The control and `0.023` arm were then rendered and scored together under the exact auditor:

| Matched exact arm | PSNR | LPIPS | SSIM | Layout PSNR | Layout SSIM |
|---|---:|---:|---:|---:|---:|
| Control | **18.1265** | 0.8187 | **0.7134** | **20.3064** | **0.8060** |
| Size `0.023` | 17.9039 | **0.7946** | 0.6901 | 19.9830 | 0.7833 |
| Intervention − control | −0.2226 | **−0.0242** | −0.0233 | −0.3234 | −0.0227 |

The smaller field is measurably more perceptually similar by LPIPS, but global coverage and layout
remain weaker. This is consistent with shrinking broad splats before the appearance path has learned
to redistribute detail into the newly exposed gaps.

## Visual evidence

- [factorized_evidence.png](factorized_evidence.png): capacity, structural gates, exact fidelity,
  and the registered size/quality Pareto bracket.
- [Original exact qualitative](audit_selected_slots20_seed0_total6000/qualitative.png): target,
  lattice, oversized irregular field, and teacher across five held-out styles.
- [Original center layout](audit_selected_slots20_seed0_total6000/field_layout.png): regular lattice
  versus learned irregular centers versus teacher, in XY and XT.
- [Matched size comparison](audit_size_control_vs_0023_seed0_600/comparison.png): exact perceptual
  and structure metrics for control and size intervention.
- [Matched qualitative](audit_size_control_vs_0023_seed0_600/qualitative.png): control and
  intervention renders shown as separate candidate columns.
- [Matched center layout](audit_size_control_vs_0023_seed0_600/field_layout.png): both candidate
  center fields shown beside lattice and teacher.

## Recommended next experiment

Do not increase fixed slots and do not merely continue the current appearance optimizer. Instead,
add **local teacher-attribute distillation**:

1. softly match active candidate centers to nearby fitted-teacher jewels;
2. supervise local log-scale, orientation, opacity mass, base color, and color gradient through that
   assignment;
3. retain the current density/sparsity/tilt gates and the factorized detached-center appearance path;
4. train a matched control versus local-distillation arm, then exact-audit before any seed expansion;
5. only after the representation reaches the exact `20 dB / 0.40 LPIPS` gate, train a small
   text-conditioned prior over its latent geometry and appearance features.

This targets the observed failure directly: the fitted teacher proves that compact irregular fields
can render at `27.70 dB`; the current encoder is failing to assign the teacher's local covariance and
appearance, not failing because irregular fields or the renderer are intrinsically inadequate.
