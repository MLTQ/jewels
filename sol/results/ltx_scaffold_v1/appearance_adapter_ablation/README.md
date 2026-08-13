# Compact scaffold-gated appearance adapter

This experiment asks whether the selected full-flow RGB correction can be replaced by a much
smaller module without giving that module control of jewel topology, lifecycle, density, geometry,
opacity, gradients, or identity. The answer is structurally yes but quantitatively not yet: the
adapter safely compresses part of the correction, but it does not meet the full teacher-replacement
gate or visibly solve the autonomous motion-field noise.

## Model and ownership contract

The adapter has **74,067 parameters**, 3.48% of the frozen 2.13M-parameter v1 flow and 28.7 times
smaller than a second full flow. Its zero-initialized head predicts only canonical RGB velocity
dimensions `(9,10,11)`. The selected top-20% scaffold-saliency gate is external to the model.

The frozen base is integrated independently from the shared Gaussian noise. After every Euler
step, topology projection, and local-to-global transform, every non-RGB candidate value is copied
from that base. The base alone owns causal row selection, continuation topology, counts, density,
stable IDs, geometry, covariance, opacity, gradients, and lifecycle. Unit and rollout tests verify
that the paired base is bit-identical to ordinary frozen sampling for the same seed.

Training and the primary visual evaluation use a **288x192** coordinate grid, preserving the
768x512 source corpus's 3:2 aspect ratio. The old 40x24 grid is used once below only to compare
against the previously selected full-flow result under its exact original metric protocol.

## Training screens

Both 3,000-step runs use 72 training views and 12 group-held-out validation views on the RTX 4090.
The direct-target run takes 230 seconds and reaches a best held-out gated RGB-velocity improvement
of 1.53% at step 2,500. It fails the renderer: the native-aspect Basketball probe loses 0.0079 dB
PSNR and 0.00254 SSIM despite slightly improving foreground color.

The teacher-distilled run takes 241 seconds. It transfers only the selected full flow's gated RGB
velocity while retaining real-video render patches as an independent target. Step 2,500 reaches a
0.60% held-out real-target improvement. At full residual strength its native Basketball probe gains
0.0130 dB and 0.00080 SSIM, but quiet temporal MAE rises `1.87e-5`, failing the declared per-class
limit. A global strength of `0.5` lowers that rise to `8.98e-6` and is the selected safe calibration.

## Exact matched regression

The table below is the original deterministic 40x24 protocol on the RTX 2070 SUPER: four validation
classes, three autonomous 16-frame windows, seed base 31 with full-validation source offsets, and
correct/shuffled/null controls. The frozen-base macro metrics are bit-identical between the old full
flow and both compact-adapter runs.

| Mechanism | PSNR | SSIM | Foreground PSNR | Foreground edge MAE | Motion-boundary MAE | Quiet temporal MAE |
|---|---:|---:|---:|---:|---:|---:|
| Frozen v1 base | 14.5700 | 0.63060 | 10.2252 | 0.68952 | 0.072349 | 0.021452 |
| Selected full-flow RGB teacher | **14.6041** | **0.63140** | **10.3591** | **0.68486** | **0.072268** | **0.021440** |
| 74k adapter, strength 1.0 | 14.5895 | 0.63085 | 10.3143 | 0.68647 | 0.072300 | 0.021441 |
| **74k adapter, strength 0.5** | 14.5798 | 0.63074 | 10.2695 | 0.68799 | 0.072323 | 0.021446 |

The selected half-strength arm gains 0.00982 dB PSNR and 0.000134 SSIM over its frozen base, raises
foreground PSNR in Basketball, HorseRiding, and PlayingGuitar, improves motion-boundary MAE in three
classes, lowers macro foreground-edge, motion, and quiet errors, and keeps every class's quiet rise
below `1e-5`. All non-RGB features, lifecycle values, count tensors, topology, density, and stable
IDs are exact for all **12** source/control rollouts.

This is not enough to replace the teacher. Half strength captures only 29% of its PSNR gain and 33%
of its foreground-PSNR gain. Full strength captures 57% and 67%, respectively, but narrowly exceeds
the matched Basketball quiet threshold (`1.021e-5`) and fails more clearly on the native grid.

## Native-aspect four-class gate

The final primary evaluation keeps the source's 3:2 aspect ratio at 288x192 and uses stable seeds
31/32/33/34. Against its independently integrated frozen base, the half-strength adapter improves
PSNR and foreground PSNR in all four classes. Macro PSNR rises by 0.00641 dB, foreground PSNR by
0.02935 dB, while foreground-edge, motion-boundary, and quiet temporal MAE fall by `9.29e-5`,
`1.02e-5`, and `5.18e-6`. Every class stays below the `1e-5` quiet-increase limit.

| Class | PSNR delta | SSIM delta | Foreground PSNR delta | Edge MAE delta | Motion MAE delta | Quiet MAE delta |
|---|---:|---:|---:|---:|---:|---:|
| Basketball | +0.00648 | +0.000395 | +0.01526 | +0.0000796 | +0.0000107 | +0.00000898 |
| HorseRiding | +0.01306 | +0.000128 | +0.09566 | -0.0004097 | -0.0000287 | -0.0000350 |
| PlayingGuitar | +0.00159 | -0.000273 | +0.00007 | +0.0000004 | -0.0000060 | +0.00000412 |
| ApplyEyeMakeup | +0.00452 | -0.000143 | +0.00643 | -0.0000418 | -0.0000167 | +0.00000121 |
| **Macro** | **+0.00641** | **+0.000027** | **+0.02935** | **-0.0000929** | **-0.0000102** | **-0.00000518** |

All non-RGB coordinates, lifecycle values, topology, and stable IDs remain exact in every native
source. This passes the adapter-versus-frozen-base structural gate, but it does not reverse the
matched teacher-replacement decision above: the visible difference is extremely small, and the
old-protocol comparison shows that the full-flow teacher retains materially more of the useful
correction.

## Protocol corrections and provenance

Filtered high-resolution runs initially exposed that seeds had been offset inside the selected
subset. The renderer now assigns every source its offset in the complete sorted validation split,
so Basketball/HorseRiding/PlayingGuitar/ApplyEyeMakeup always use seeds 31/32/33/34 regardless of
filtering. Each source record stores its realized seed. Reports also store the physical GPU name and
compute capability because deterministic CUDA kernels do not imply bit-identical free-running
fields across Turing and Ada GPUs. Candidate-versus-base deltas are therefore always device-local.

## Decision

Keep the adapter and exact-ownership rollout as useful infrastructure, but **do not select this
checkpoint as a replacement for the full-flow correction and do not expand its mutable dimensions**.
The direct-target loss is insufficient; one-step teacher-velocity distillation only partly follows
the teacher after autonomous integration. The narrow follow-up is complete-trajectory distillation
with generated appearance carry (`jewels-pz2`).

More importantly, the native contacts show that frozen and RGB-adapted fields are nearly
indistinguishable beside the fitted ceiling. The dominant remaining error is not an RGB-only
capacity deficit. New jewels are addressed by cell and rank but lack a shared local set/trajectory
state, producing blurry structured speckle around actors and thin moving objects. The primary
architectural follow-up is a cell- or neighborhood-coupled birth-set realizer (`jewels-2qb`) before
direct text-to-jewel distillation.

## Artifacts

- `direct_train_summary.json` and `direct_train_log.jsonl` retain the rejected direct-target run.
- `distilled_train_summary.json` and `distilled_train_log.jsonl` retain teacher-distilled training.
- `distilled_four_class_{half,full}_strength_matched_summary.json` are the exact 12-control
  old-protocol comparisons.
- `distilled_four_class_half_strength_native_summary.json` combines the definitive four-source
  native-aspect deltas and invariant gates; its two source summaries retain the complete records.
- `direct_basketball_probe_summary.json`, `distilled_basketball_{full,half}_strength_summary.json`,
  and `full_flow_basketball_top20_summary.json` retain native-aspect diagnostic screens.
- `basketball_half_strength_contact.png`, `horse_guitar_eye_half_strength_contact.png`, and the four
  selected `*_half_strength_288x192.gif` files retain the primary native-aspect visual evidence.
- Checkpoints remain on Aine under
  `/home/m/jewels/topology/scaffold_appearance_adapter_v1/{run_3000_seed41,distill_3000_seed43}`.
