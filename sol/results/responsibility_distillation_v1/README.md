# Renderer-responsibility distillation v1

## Outcome

Renderer-mediated teacher moments produce a real fidelity signal, but the registered experiment does
**not** pass the joint mechanism gate or the absolute promotion gate. The response-appearance arm
improves exact PSNR by `0.05865 dB`, SSIM by `0.00372`, and layout PSNR by `0.08994 dB` while exact
LPIPS is effectively tied but `0.00018` worse. Adding responsibility geometry increases the PSNR
gain to `0.09360 dB` but worsens LPIPS by `0.00697`.

A preregistered bridge between responsibility and position-only appearance nearly closes the
tradeoff. Its single midpoint improves exact LPIPS by `0.00487` (`0.64%`) and loses only
`0.00630 dB` PSNR. It improves PSNR in three of five held-out styles and LPIPS in four of five, but
the macro PSNR sign is negative, so it is not called a win and is not replicated at seeds 1 and 2.

The negative claim is deliberately narrow: seed 0, 600 continuation steps, one teacher sample size,
one support kernel, the registered weights, and the v3 bounded appearance head. The positive result
is also narrow: two independently motivated objectives bracket a joint exact improvement very
closely while all irregularity, sparsity, and time-distortion gates remain intact.

## Intervention

For every detached student center, the fitted teacher jewels are weighted by optical mass and their
Mahalanobis support. The target covariance contains both within-teacher covariance and the spread of
contributing teacher centers. Color is evaluated locally at the student center and its analytic
mixture Jacobian is retained. Queries without a sampled in-support teacher use the minimum-
Mahalanobis teacher and are counted rather than silently producing an empty target.

All arms continue the same 40,960-proposal factorized-v3 checkpoint with identical data, seed,
optimizer, renderer, and structural objectives. Responsibility targets use 4,000 active-uniform
teacher jewels, 1,024 sampled student jewels, five-sigma support, temperature `1.0`, and an
independent CPU generator so enabling the loss does not perturb the ordinary teacher or GPU sample
sequence. `PROTOCOL.md`, `PROTOCOL_HYBRID.md`, and `PROTOCOL_HYBRID_MIDPOINT.md` were frozen before
their respective optimization runs.

## Read-only calibration

`calibration.json` measures each raw objective and its parameter-gradient norm without taking an
optimizer step:

| Component | Raw loss | Geometry gradient | Appearance gradient |
|---|---:|---:|---:|
| sampled render | 0.01914 | 0.17221 | 0.01294 |
| response scale | 0.63015 | 3.68123 | 0 |
| response axis | 0.36218 | 0.27232 | 0 |
| response opacity | 0.45941 | 0.80975 | 0 |
| response color | 0.10104 | 0 | 0.20771 |
| response Jacobian | 0.03010 | 0 | 0.02924 |

The sampled target has 6.52 supported candidates and 1.73 effective contributors per student on
average. Only `0.586%` of queries use fallback. No scale targets require projection, but `44.1%` of
color elements and `75.5%` of Jacobian elements fall outside the v3 head's feasible bounds before
projection. That clipping is the strongest measured explanation for why raw local targets pull
LPIPS one way while feasible responsibility targets pull PSNR the other.

## Registered responsibility screen

| Arm | Sampled PSNR | Delta | Occupancy | Active | Mixed tilt | Eligible |
|---|---:|---:|---:|---:|---:|:---:|
| control | 18.7007 | — | 0.98430 | 0.63116 | 0.52101 | reference |
| response-appearance | 18.7124 | +0.0117 | 0.98443 | 0.63157 | 0.52061 | yes |
| response-full | 18.7517 | +0.0510 | 0.98439 | 0.63541 | 0.51880 | yes |

Both interventions remain below occupancy `0.985`, active fraction `0.70`, and the `0.50 dB`
screen-loss limit, while mixed spacetime tilt remains far above `0.25`.

## Exact responsibility audit

| Arm | PSNR | Delta | LPIPS | LPIPS change | SSIM | Layout PSNR | Joint win? |
|---|---:|---:|---:|---:|---:|---:|:---:|
| control | 18.50932 | — | 0.75795 | — | 0.75547 | 20.98735 | reference |
| response-appearance | 18.56796 | +0.05865 | 0.75813 | +0.00018 | 0.75919 | 21.07729 | no |
| response-full | 18.60292 | +0.09360 | 0.76493 | +0.00697 | 0.76110 | 21.10555 | no |

Response-appearance improves PSNR in four of five styles and LPIPS in three of five. Response-full
improves PSNR in four of five but worsens LPIPS in all five. Neither passes the registered
requirement that one macro arm improve both exact metrics.

## Preregistered bridge tests

The half-strength hybrid keeps response RGB/Jacobian at `0.025 / 0.025` and adds position-only raw
local RGB/gradient at `0.05 / 0.10`. It reverses the perceptual sign but overshoots the fidelity
margin. The single midpoint halves those raw-local weights once; the protocol explicitly forbids
further interpolation.

| Arm | Sampled PSNR | Exact PSNR | PSNR delta | Exact LPIPS | LPIPS improvement | Joint win? |
|---|---:|---:|---:|---:|---:|:---:|
| matched control | 18.70072 | 18.50932 | — | 0.75795 | — | reference |
| half-strength hybrid | 18.53137 | 18.47671 | -0.03261 | 0.75431 | 0.00364 | no |
| midpoint hybrid | 18.63590 | 18.50301 | -0.00630 | 0.75308 | 0.00487 | no |

The midpoint remains structurally eligible: exact occupancy is `0.98486`, active fraction is
`0.63504`, and median mixed spacetime tilt is `0.52022`. By style, its exact deltas are:

| Style | PSNR delta (dB) | LPIPS improvement |
|---|---:|---:|
| anime | +0.11764 | -0.00061 |
| cartoon | -0.29093 | +0.01129 |
| clay | +0.05629 | +0.00831 |
| photoreal | -0.05574 | +0.00318 |
| render3d | +0.14121 | +0.00220 |

## Visual evidence

- `evidence.png` graphs the registered screen, exact metrics, structure gates, and per-style deltas.
- `audit_control_appearance_full_seed0_600/qualitative.png` is the exact-render contact sheet for
  control, response-appearance, response-full, target, lattice baseline, and fitted teacher.
- `audit_control_appearance_full_seed0_600/field_layout.png` directly plots active jewel centers in
  XY and X–time slices. The learned candidates are irregular; they do not occupy the proposal grid.
- `audit_control_vs_hybrid_midpoint_seed0_600/qualitative.png` compares the exact control and frozen
  midpoint renders.
- The three audit directories include `comparison.png` graphs and source-owned `report.json` data.

The contact sheets show a limitation that the small metric differences should not obscure. Learned
irregular renders remain broad, soft, and glitter-like; object boundaries are largely absent and the
responsibility variants are visually subtle. The teacher and even the lattice reconstruction remain
far clearer. This is no longer quantization noise from jewel centers: it is an appearance/capacity
and overlapping-kernel problem in the learned irregular field.

## Feasibility and next gate

This is stronger evidence for the mechanism than the earlier position-only result: renderer-
mediated targets recover PSNR/layout while raw local appearance recovers LPIPS, and a frozen
midpoint lands within `0.00630 dB` of a joint exact win without weakening the irregular field. It
makes a more-compute scaling argument plausible, but it is **not yet conclusive text-to-video proof**.
The best candidate remains at `18.50 dB / 0.753 LPIPS`, far from the absolute `20 dB / 0.40` gate
and farther from the fitted teacher at about `27.70 dB / 0.172`.

More scalar weight search is not warranted by this tranche. The next experiment should change the
bounded appearance contract that clips most responsibility targets: predict a renderer-compatible
unconstrained/residual color basis or supervise composited local color and Jacobian samples without
projecting them into per-jewel sigmoid RGB and `0.25*tanh` gradients. The next gate should require:

1. materially lower target saturation under a read-only calibration;
2. an exact macro PSNR and LPIPS win over both the matched control and this midpoint;
3. unchanged occupancy, active-fraction, and mixed-tilt gates;
4. visible boundary or contrast recovery in the same source-owned contact sheet;
5. only then, longer training and seeds 1–2 before any pitch claim.

The responsibility computation is currently quadratic in sampled students and teachers. Before a
larger compute run it should be tiled or spatially indexed so increased teacher support buys target
quality rather than avoidable memory and runtime cost. Runtime from this experiment is not reported
as a model property because the shared GPU was concurrently loaded.
