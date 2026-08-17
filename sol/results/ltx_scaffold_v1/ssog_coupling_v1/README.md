# SSOG birth-set coupling: steered separable Gaussian fields vs convolution

Tests whether an SSOG-style attention field (Pisoni 2026: Gaussian atoms over relative
position, content steering through bounded cold-started residuals, separable application)
outperforms the 3x3x3 convolution mixing of the retained coupled-set direction. Motivation: the
layout-divergence diagnosis made long-range macro-structure the gating deficit, and trained
SSOG atoms form exactly such long-range geometry. Implementation:
`sol/birth_set_coupling.py::SsogBirthSetBlock` behind `BirthMarkFlowModel(set_coupling="ssog")`;
occupancy rides the same separable passes as the state so the field averages over born cells
only (a required departure from the ViT setting, where every grid position holds a token).
Exact 2qb harness: augment-from run_12000, frozen base, 3,000 steps, seed 47, native 192x288
guide preparation, zero-residual start (bit-identical to base at init, unit-verified).

## Screen (held-out velocity, same-device battery — see methods note below)

| Arm | Added params | Correct loss | Δ vs matched base |
|---|---:|---:|---:|
| Frozen base (2070S battery) | — | 1.6312 | — |
| Conv-coupled step2250 (re-evaluated on 2070S) | 414,016 | 1.6280 | −0.20% |
| **SSOG step3000** | **66,367** | **1.6236** | **−0.47%** |

2.4x the conv screen gain at 6.2x fewer parameters. Screens select checkpoints only.

## Rendered gates (single seed 31, native 288x192)

Free three-window rollout (`perceptual_report.json`, all arms rendered in one report):
SSOG improves the frozen base on LPIPS (0.6901 vs 0.6971), PSNR (14.289 vs 14.142) and layout
PSNR (15.511 vs 15.330, three of four classes) but regresses SSIM (0.5887 vs 0.6001); against
the conv arm it ties LPIPS and trails PSNR/SSIM/layout. At the measured seed-noise floor none
of these rendered deltas are individually decisive.

Paired exact-count audit (`paired_summary.json`, base owns counts/ranks/IDs):

| Δ vs frozen base under identical counts | SSOG (66k) | Conv reference (414k) |
|---|---:|---:|
| PSNR | +0.224 (3/4 classes) | +0.333 (4/4) |
| Foreground PSNR | **+0.775** | +0.761 |
| Foreground edge / motion / quiet MAE | all improved | all improved |
| SSIM | −0.017 (0/4 classes) | −0.001 (2/4) |

## Learned geometry (`atom_geometry.json`)

Three of four atoms look backward in time (mu_t = −1.0/−1.3/−0.2 cells) with temporal reach
(sigma_t 1.7–1.8) wider than spatial (~1.3): the field discovered temporal-context gathering —
condition new births on where jewels were just born. All three cold-started gates opened
(mu 0.20, sigma 0.52, lambda 0.13 from 0.01), sigma hardest: content chiefly modulates reach.

## Decision

Not selected, same status as the conv checkpoint: both couplings deliver real foreground/detail
gains under exact counts and fail the SSIM/visual criteria of `jewels-dj2`, whose rendered
structural objective remains the open lever. The SSOG field is retained as the preferred
coupling substrate for that work: equal-or-better exact-count gains at 16% of the parameters,
interpretable geometry, and reach that a local convolution cannot express. Alternative
still untested: applying the field at the jewel level (jewel-to-jewel Gaussian affinity is
closed-form in this representation) rather than the cell level.

## Methods note — evaluation batteries are device-bound

The fixed seed-47 evaluation paths differ across GPU architectures (same seed, different CUDA
RNG sequences): run_12000's base scores 1.541898 on the 4090 battery and 1.631154 on the 2070S
battery. The ~0.09 offset dwarfs model deltas, so arms are only comparable within one device;
every comparison above is same-device, and all previously committed comparisons ran uniformly
on the 4090 and stand.
