# LPIPS-versus-layout divergence: diagnosis

The domain-matched realizer improved LPIPS while its contact sheets visibly lost macro-layout.
This record adds a layout-sensitive metric to the battery, tests the per-view overexposure
hypothesis with a matched-exposure retrain, and tests corpus composition with a mixed
24-source stack. Protocol identical throughout (frozen recipe, seeds 23/17, deterministic
seed-31 native 288x192 correct-only rollouts, shared reference arms reproduce bit-consistently
in every report).

## The layout metric

`sol/perceptual_eval.py::layout_signature` average-pools texture away (factor 8: 36x24 cells at
native resolution) and scores PSNR/SSIM on the pooled videos — where things are, not how they
are textured. It is now part of every `perceptual-arm-eval-v1` report.

## Results (macro over the four held-out scaffolds)

| Arm | Train corpus | Views | Velocity | LPIPS | Layout PSNR | Layout SSIM |
|---|---|---:|---:|---:|---:|---:|
| Guide upsample baseline | — | — | — | 0.6761 | 22.608 | 0.9286 |
| Fitted jewel ceiling | — | — | — | 0.0982 | 30.457 | 0.9912 |
| Generated, UCF (v1 base) | 12 real UCF | 72 | 1.5582 | 0.6971 | **15.330** | 0.6662 |
| Generated, coupled | 12 real UCF | 72 | — | 0.6911 | 15.748 | 0.6710 |
| Generated, mixed | 12 UCF + 12 LTX | 108 | 1.1782 | 0.6862 | 14.188 | 0.6323 |
| Generated, LTX domain 12k | 12 LTX | 36 | **1.0448** | **0.6760** | 13.380 | 0.6319 |
| Generated, LTX domain 6k (matched exposure) | 12 LTX | 36 | 1.0757 | 0.6698 | 12.895 | 0.5777 |

## Findings

1. **The metric captures what the eyes saw.** Layout PSNR ranks UCF > mixed > domain — the
   exact reverse of LPIPS — quantifying the ~2 dB macro-structure gap that patch-based LPIPS
   missed. The battery now reports texture (LPIPS) and structure (layout) separately.
2. **Overexposure is ruled out.** Matching the UCF recipe's 167 steps/view (6,000 steps)
   worsens layout further (13.380 to 12.895) while nudging LPIPS down; the divergence is not a
   training-schedule artifact.
3. **Corpus composition is causal, and the corpora are complementary.** The mixed stack
   interpolates between the pure corpora on every axis — no synergy, no interference. LTX fits
   supply texture/feature accuracy (velocity 1.045 vs 1.558); real-video fields supply layout
   structure. One hypothesis for why fits behave differently across domains (jewel-statistics
   differences between LTX and UCF fits) remains open.
4. **Every generated arm sits far below the baseline on layout** (best 15.7 vs 22.6): the
   macro-structure crossover, not the texture crossover, is now the milestone the scaling
   program should gate on — track layout PSNR as the structure metric per corpus doubling.

## Consequence for the scaling program

Corpus growth on the teacher axis should keep real-video fields in the mixture and gate on
layout PSNR + LPIPS jointly (plus the rendered audit). The scaling curve's LPIPS slope stands;
its layout column starts with this record.

Artifacts: three perceptual reports (12-source arms rescored with layout; domain-6000;
mixed-24-source), both new training summaries, and the mixed rollout contact sheet. Aine:
`topology/ltx_mixed_v1`, `topology/ltx_domain_v1/{mark_6000_matched_exposure,rollout_6000_*,
perceptual_6000}`, `eval/perceptual_native_v2`.
