# Structured tokenizer spike results

> This file records the earlier 6,471-jewel control. The current density-corrected work is in
> [`dense/README.md`](dense/README.md), and the hierarchy/generation/editing result is in
> [`axial/README.md`](axial/README.md). Cross-domain UCF evidence is in
> [`transfer/README.md`](transfer/README.md).

Prompted persistent-streaming preparation and held-out text geometry are recorded in
[`prompt_smoke/README.md`](prompt_smoke/README.md).
The first direct-jewel correct/shuffled/null prompt experiment and free-count videos are in
[`prompt_smoke/direct_prompted_streaming_3000/README.md`](prompt_smoke/direct_prompted_streaming_3000/README.md).
That record now also contains the exact-topology washout decomposition, an oracle-topology
stochastic mark-flow control, and the decisive low-resolution oracle-video-guide experiment that
selects a semantic-scaffold-to-jewel architecture over a count-head-only fix.
The first real prompt-generated replacement scaffold corpus and its four class-level visual audits
are in [`ltx_scaffold_v1/README.md`](ltx_scaffold_v1/README.md). Its four density-matched 72k jewel
fits and completed leakage-safe UCF-train/LTX-validation realizer gate live beneath that folder.
The completed matched low-texture domain test is in
[`ltx_cel_eval_v1/README.md`](ltx_cel_eval_v1/README.md). Flat-region error falls 35.7%, but
contour-region error rises 62.3% at effectively identical density, selecting an explicit
fill/contour allocation test. A subsequent user-directed
[`same-field generator adaptation`](ltx_cel_eval_v1/generator_adaptation/README.md) transfers the
selected UCF topology/mark stack to all four styled fields, with source overlap explicitly recorded
and no claim of unseen generalization. Its fitted-seed-free 48-frame rollouts reach 15.401 dB /
0.4270 SSIM under the correct scaffold versus 12.642 / 0.0571 shuffled; density is already matched,
while excessive temporal noise selects appearance/stability as the next generator bottleneck.
The first multiscale token-guide/render-loss control is recorded under
[`prompt_smoke/direct_prompted_streaming_3000/mark_flow_multiscale_render2_12000/README.md`](prompt_smoke/direct_prompted_streaming_3000/mark_flow_multiscale_render2_12000/README.md);
it improves contrast/edge energy but is not selected because PSNR, SSIM, saturation, and temporal
stability do not improve together. The capacity-matched
[`raster-plus-token control`](prompt_smoke/direct_prompted_streaming_3000/mark_flow_hybrid_feature_12000/README.md)
also fails, so the original cross-cell/global raster conditioner remains selected while render
supervision is tested as an isolated objective change. Raster-guide render weights
[`2.0`](prompt_smoke/direct_prompted_streaming_3000/mark_flow_raster_render2_12000/README.md) and
[`0.5`](prompt_smoke/direct_prompted_streaming_3000/mark_flow_raster_render05_12000/README.md) are
also non-dominating trades: the lower weight recovers SSIM/edge detail but destabilizes Guitar. The
selected v1 model subsequently passes cross-domain LTX video-scaffold realization at 15.342 dB /
0.6942 SSIM against the 13.777 dB / 0.4160 deterministic control. This is a privileged
video-guide/topology result, not free prompt-only jewel generation. The follow-up
[`learned-topology gate`](ltx_scaffold_v1/topology_eval/README.md) removes fitted cells/counts/ranks
for one continuation stride: frozen-flow rendering reaches 15.464 dB / 0.6967 SSIM and 5,822
effective contributors/frame, within 0.028 dB of oracle topology with exact carry. Initial and
multi-stride generated state remain open.

## Decision

The deterministic count-aware raster tokenizer is feasible on the 6,471-jewel Avenue corpus and is
good enough to unlock a latent-prior spike. The render-aware objective is load-bearing; structural
feature/count losses alone are not a sufficient proxy for video appearance.

## Protocol

- Date: 2026-08-04
- GPU: allocated RTX 2070 SUPER (`cuda:1`, 8 GB)
- Corpus: 231 fitted windows; 214 training and 17 validation windows
- Leakage control: whole source videos `11` and `15` held out
- Grid: `12×12×6`, 80 slots per cell; observed maximum occupancy 67
- Model: 1.09M parameters, 32 values per cell, 27,648 latent values
- Raw fitted window: 6,471 × 22 = 142,362 values; 5.15× numeric bottleneck
- Metric: exact all-jewel renders at fixed uniform continuous `(u,v,t)` samples, target fitted field
  versus decoded fitted field

## Controlled 1,000-step ablation

| Render weight | Mean held-out PSNR | Mean count ratio | Runtime |
|---:|---:|---:|---:|
| 0.0 | 11.719 dB | 0.9782 | 24.6 s |
| 0.1 | 13.285 dB | 0.9687 | 32.1 s |
| 0.5 | 15.145 dB | 0.9326 | 31.8 s |

All arms used the same seed, architecture, source split, four validation windows, and 256 render
points per window. Weight 0.5 improved the held-out score by 3.43 dB over the structural-only arm.

## Selected 5,000-step run

Weight 0.5 with 32 fresh render samples per step reached **17.405 dB mean held-out PSNR** and
**0.99745 mean count ratio** at step 5,000 in 154.6 seconds. The embedded evaluation used four
source-balanced validation windows and 512 points per window. Intermediate mean PSNR was 13.753 at
step 1,000, 15.665 at 2,000, 16.794 at 3,000, and 17.168 at 4,000.

A post-training audit over **all 17 held-out windows** with 2,048 render points each measured 17.431
dB window-weighted mean, 17.684 dB median, and **17.192 dB macro-by-source mean**. Source means were
17.771 dB for source `11` and 16.614 dB for source `15`; the macro score is the preferred comparison
because the sources contribute 12 and five windows respectively. The full per-window record is in
`selected_heldout_eval.json`.

The completed checkpoint remains on the training host at
`/home/m/jewels/tokenizer/sol_render_w05_5000/autoencoder.pt`. The adjacent JSONL log and summary are
copied into this directory.

## Interpretation and limits

- This proves that a deterministic editable raster bottleneck can reconstruct unseen fitted jewel
  fields and that the allocated 2070S is sufficient for iteration.
- The 17.405 dB score is 1.26 dB above the older 16.15 dB reported stochastic-tokenizer result, but
  that comparison is directional rather than apples-to-apples: the older run did not use this
  source-held-out sampled-render protocol.
- The all-window sampled audit is stable, but rendered videos and temporal/perceptual metrics still
  need visual inspection before the tokenizer is frozen.
- This metric isolates tokenizer error relative to fitted renders; it does not include the fitter's
  error relative to source pixels.
- The 45k-jewel dense corpus does not fit this decoder: measured 12×12×6 occupancy reaches 399. It
  needs sparse or hierarchical decoding rather than a simple slot-count increase.

## Next gate

1. Render held-out error videos for visual and temporal inspection.
2. Cache frozen encoder latents and their existing CLIP sidecars.
3. Train the text-conditioned raster flow against scene-mean and nearest-window baselines.
4. Exercise the same flow through dirty-cell clamped inpainting while injecting moved-jewel
   constraints.
