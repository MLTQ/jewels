# Residual appearance contract v1

## Outcome

Removing the factorized encoder's per-jewel appearance bounds produced the largest verified
irregular-encoder gain so far. A zero-initialized residual head raises exact seed-0 PSNR from the
bounded matched control's `18.5093` to `19.3555 dB`, lowers LPIPS from `0.75795` to `0.74755`, and
raises SSIM from `0.75547` to `0.80897`, without restoring the jewel lattice. One equal continuation
reaches `19.6648 dB / 0.73288` LPIPS and improves both exact metrics in all five held-out styles.

This passes the registered seed-0 **representation mechanism** rule. It does not pass the absolute
`20 dB / 0.40 LPIPS` promotion rule or the strict three-seed structure replication. Seeds 1 and 2
repeat near-20 dB sampled fidelity but finish just above the `0.985` occupancy-uniformity limit.
The defensible conclusion is therefore: the bounded appearance head was a major causal bottleneck,
and additional compute continues to improve the residual representation, but geometry must be held
or stabilized across seeds and a stronger perceptual objective is still required before a text-to-
video feasibility pitch.

## Intervention

`FactorizedStructuralJewelEncoder` retains its existing bounded prediction and adds a 12-output
unconstrained residual for RGB and RGB Jacobians. The new head is initialized to zero. A checked
transfer requires every old parameter to load and allows only the two residual-head tensors to be
missing, so the expanded model's initial predictions are bitwise identical to its source.

The renderer already accepts unbounded fitted colors and P1 gradients. Responsibility loss can now
score raw composited color/Jacobian moments instead of projecting them to RGB `[0,1]` and gradient
`[-0.25,0.25]`. Geometry, covariance, opacity, background, canonical 22-D features, and renderer
semantics are unchanged.

## Read-only calibration

The frozen seed-0 calibration uses 4,000 active-uniform fitted teachers, 1,024 opacity-sampled
students, five-sigma support, and 1,024 render points without an optimizer:

| Component | Raw loss | Geometry gradient | Appearance gradient |
|---|---:|---:|---:|
| sampled render | 0.01914 | 0.17221 | 0.06938 |
| raw response color | 0.14491 | 0 | 1.33007 |
| raw response Jacobian | 1.90608 | 0 | 0.10480 |

Support statistics reproduce responsibility v1: 6.52 supported and 1.73 effective teachers per
student, with `0.586%` fallback. `44.1%` of color and `75.5%` of Jacobian elements exceed the old
bounds, but the residual/raw contract projects `0%`. Registered response weights remain
`0.025 / 0.025`; their relative gradient pressure is close to the earlier bounded arm rather than
being selected from training outcomes.

## Registered 600-step screen

| Arm | Sampled PSNR | Occupancy | Active | Mixed tilt | Eligible |
|---|---:|---:|---:|---:|:---:|
| bounded control | 18.7007 | 0.98430 | 0.63116 | 0.52101 | reference |
| bounded midpoint | 18.6359 | 0.98460 | 0.63504 | 0.52022 | reference |
| residual-control | 19.5402 | 0.98389 | 0.62941 | 0.52293 | yes |
| raw-response | 19.5564 | 0.98438 | 0.63288 | 0.52111 | yes |

The new head alone adds `0.8395 dB` sampled PSNR over the bounded control. Raw responsibility adds a
further `0.0162 dB`. Both retain the frozen structure thresholds.

## Registered exact audit

| Arm | PSNR | Delta vs bounded | LPIPS | Improvement vs bounded | SSIM | Layout PSNR |
|---|---:|---:|---:|---:|---:|---:|
| bounded control | 18.50932 | — | 0.75795 | — | 0.75547 | 20.98735 |
| bounded midpoint | 18.50301 | -0.00630 | 0.75308 | 0.00487 | 0.75517 | 20.97977 |
| residual-control | 19.35552 | +0.84621 | 0.74755 | 0.01041 | 0.80897 | 22.41839 |
| raw-response | 19.37601 | +0.86670 | 0.74696 | 0.01099 | 0.80900 | 22.45724 |

Residual-control improves PSNR in all five styles and LPIPS in four; anime LPIPS is effectively tied
but `0.00044` worse. Raw-response narrowly beats its matched residual-control in both macro metrics:
`+0.02049 dB` PSNR and `0.00058` lower LPIPS. That causal add-on is heterogeneous:

| Style | Raw-response PSNR delta | LPIPS improvement |
|---|---:|---:|
| anime | +0.02512 | -0.00046 |
| cartoon | -0.07480 | +0.00015 |
| clay | -0.02858 | -0.00040 |
| photoreal | +0.00935 | -0.00446 |
| render3d | +0.17138 | +0.00808 |

Only render3d is a per-style joint response win. The macro causal pass is recorded, but the strong
claim belongs to the residual representation rather than raw responsibility.

## Equal continuation

The eligible residual paths received one preregistered equal 600-step continuation. Continued raw-
response reached sampled `19.8892 dB` but occupancy `0.98517`, so it failed the screen and was not
exact-audited. Continued residual-control remained eligible:

| Residual-control state | Sampled PSNR | Occupancy | Active | Mixed tilt |
|---|---:|---:|---:|---:|
| step 600 | 19.5402 | 0.98359 | 0.62941 | 0.52293 |
| total step 1,200 | 19.9045 | 0.98458 | 0.62815 | 0.52115 |

| Residual-control state | Exact PSNR | Exact LPIPS | SSIM | Layout PSNR |
|---|---:|---:|---:|---:|
| step 600 | 19.35551 | 0.74760 | 0.80897 | 22.41838 |
| total step 1,200 | 19.66478 | 0.73288 | 0.82256 | 23.04958 |
| improvement | +0.30926 | 0.01473 lower | +0.01360 | +0.63120 |

The continuation improves PSNR and LPIPS in all five styles. PSNR gains range `0.1700–0.6160 dB`;
LPIPS improvements range `0.00987–0.02295`. Target-relative contrast rises `0.8046 -> 0.8108`, edge
ratio `0.3565 -> 0.3786`, and saturation `0.9163 -> 0.9382`. Temporal-change ratio moves farther
above one (`1.5548 -> 1.6681`), consistent with the remaining glitter/speckle instability.

## Frozen seed replication

Seeds 1 and 2 repeat the two 600-step residual-control tranches from the same upstream bounded
checkpoint. These are continuation-sampling seeds, not independent upstream encoder initializations.

| Seed | Sampled PSNR | Occupancy | Active | Mixed tilt | Eligible for exact? |
|---|---:|---:|---:|---:|:---:|
| 0 | 19.9045 | 0.98458 | 0.62815 | 0.52115 | yes |
| 1 | 19.8993 | 0.98571 | 0.63552 | 0.52487 | no |
| 2 | 19.9454 | 0.98664 | 0.63674 | 0.52148 | no |

Fidelity, sparsity, and time distortion repeat tightly, but the center distribution drifts toward
uniform coverage in seeds 1 and 2 by `0.00071` and `0.00164` beyond the threshold. Per protocol they
were not exact-audited, and three-seed promotion fails. This is a scoped structural-stability result,
not evidence that the large appearance gain failed to repeat.

## Visual evidence

- `evidence.png`: bounded frontier versus the two registered residual arms, exact metrics, structure,
  and per-style raw-response deltas.
- `audit_control_response_seed0_600/qualitative.png`: matched step-600 exact contact sheet.
- `audit_control_response_seed0_600/field_layout.png`: direct XY and X–time center plots; both
  candidates are irregular, not quantized to the proposal lattice.
- `continuation_2070/audit_control600_vs_control1200/qualitative.png`: step-600 versus total-1,200
  residual-control contact sheet.
- `continuation_2070/audit_control600_vs_control1200/comparison.png`: exact continuation metrics and
  structural gates.
- `replication_screen.png`: three-seed sampled fidelity and structure thresholds.
- Every audit directory contains its source-owned `report.json`.

The residual sheet has visibly stronger dark/light separation and more recognizable coarse people,
hands, counters, and objects than the bounded sheet. Continuation strengthens these cues. It does not
restore fine boundaries, and isolated dark speckles become more visible. The lattice and fitted
teacher remain substantially clearer.

## Feasibility assessment and next gate

The result materially strengthens the compute-feasibility case for time-distorted Gaussian splats.
Exact seed-0 PSNR gains `1.1555 dB` over the bounded control, and an equal continuation improves both
metrics in every style rather than trading one for the other. The best state is only `0.3352 dB`
below the PSNR gate. More compute may plausibly cross `20 dB`.

Compute alone is not a credible route from `0.7329` to `0.40` LPIPS: the fitted ceiling is
`27.699 / 0.172`, and the frozen lattice is `23.652 / 0.428`. The remaining work is perceptual and
structural, not merely scale. The next experiment should freeze the already-passing geometry branch
while training residual appearance, which makes occupancy invariant by construction, and compare
render-only appearance against a bounded full-frame multiscale/edge objective. It should also
penalize rare out-of-range rendered pixels or residual energy to reduce dark speckles without
restoring per-jewel clipping.

The next gate should require all three continuation seeds to retain the exact source geometry,
improve exact PSNR and LPIPS over bounded control, show lower temporal overshoot/speckling, and reach
mean PSNR `>=20 dB`. Only after that should the text prior be trained against this representation and
evaluated for prompt-conditioned motion/appearance control.

## Execution integrity

`EXECUTION_NOTE.md` records the machine's reversed CUDA/nvidia-smi ordering. Initial matched arms ran
on the shared 4090; official continuation and replication used the explicitly mapped 2070. An
incorrect validation launch was stopped before optimization, a partial wrong-device continuation
produced no checkpoint and was excluded, and the 2070 llama service was runtime-masked during the
official tranche then restored successfully. Runtime is not used as model evidence.
