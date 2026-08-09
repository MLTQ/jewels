# Structured tokenizer spike results

> This file records the earlier 6,471-jewel control. The current density-corrected work is in
> [`dense/README.md`](dense/README.md), and the hierarchy/generation/editing result is in
> [`axial/README.md`](axial/README.md). Cross-domain UCF evidence is in
> [`transfer/README.md`](transfer/README.md).

Prompted persistent-streaming preparation and held-out text geometry are recorded in
[`prompt_smoke/README.md`](prompt_smoke/README.md).
The first direct-jewel correct/shuffled/null prompt experiment and free-count videos are in
[`prompt_smoke/direct_prompted_streaming_3000/README.md`](prompt_smoke/direct_prompted_streaming_3000/README.md).

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
