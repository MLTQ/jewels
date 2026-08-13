# Guide-upsample baseline: trivial decode of the rollout's own scaffold input

## Question

The autonomous three-window rollout receives, per 16-frame stride, exactly one pixel input:
the `(16,16,8)` cell-RGB guide computed from the LTX evaluation video already resized to
48x80. How does trivially trilinear-upsampling those same guides back to 48x80 score under
the identical rollout protocol — before any learned topology, marks, or rendering?

## Protocol

`sol/guide_upsample_baseline.py`, joined against the deterministic seed-31 rollout report
(`../scaffold_mark_rollout/summary.json`). Per stride at frontiers 0/16/32:
`video_to_cell_raster(video[f:f+16], GridSpec((16,16,8)))`, inverted and trilinearly
upsampled to `(16,48,80,3)`, concatenated to 48 frames, scored with the same
`render_signature` / `saliency_render_signature` (fitted-reference background) /
`_seam_report` against the same resized target slice. The baseline sees nothing the
generation stack does not see; it sees *less* (no text embedding, no carried state).

```
python -m sol.guide_upsample_baseline \
  --manifest <scaffold_topology_v1 manifest> --video-root <ltx eval mp4 dir> \
  --rollout-summary sol/results/ltx_scaffold_v1/scaffold_mark_rollout/summary.json \
  --out sol/results/ltx_scaffold_v1/guide_upsample_baseline
```

## Results (48 frames, four held-out LTX evaluation scaffolds)

| Arm | PSNR | SSIM | edge ratio | temporal-change ratio | fg PSNR | quiet MAE |
|---|---:|---:|---:|---:|---:|---:|
| Guide upsample (this baseline) | **21.081** | **0.9037** | 0.515 | 0.508 | **16.54** | **0.00643** |
| Generated correct (seed 31) | 14.570 | 0.6306 | **1.205** | **1.192** | 10.23 | 0.02145 |
| Fitted jewel ceiling | 21.840 | 0.9310 | 1.449 | 1.887 | — | — |

Per class, baseline vs generated-correct PSNR: Basketball 21.714 vs 16.235; HorseRiding
21.003 vs 15.167; PlayingGuitar 20.003 vs 13.082; ApplyEyeMakeup 21.603 vs 13.796. The
baseline wins every class by 5.5-7.8 dB and also wins foreground PSNR in every class. Its
stride-boundary seams are real (seam-to-regular 1.6-3.0) but comparable to target seam level
(seam-to-target 1.09 mean).

## Reading

1. **Every pixel-fidelity metric prefers the trivial decode.** The blurred upsample sits
   0.76 dB below the *fitted 72k-jewel ceiling* on PSNR while the learned stack sits 7.3 dB
   below it. Single-reference PSNR/SSIM cannot demonstrate the generative stack's value.
2. **What the learned stack measurably adds is detail and motion energy.** The baseline
   carries half the target's edge energy (0.515) and half its temporal change (0.508); the
   generated field restores both to ~1.2 — visible texture and motion at roughly target
   energy, misplaced enough to lose PSNR. This is the distortion-perception trade stated
   quantitatively.
3. **Temporal stability is currently a baseline win, not a field win.** Quiet-region
   temporal error is 3.3x lower for the blurry decode than for the generated field —
   consistent with the appearance/stability bottleneck identified by the lifecycle
   factorization and cel-adaptation audits.
4. Any future claim of rollout quality must beat, or explicitly decline to compete with,
   this baseline — and the paper's evaluation must add perceptual/distribution metrics
   (LPIPS/FVD-class) where "restores detail energy" can actually score.

Artifacts: `report.json` (schema `guide-upsample-baseline-v1`, per-class render/saliency/seam
signatures plus joined pipeline numbers), `contact_sheet.png` (target row over baseline row,
frames 0/15/16/47 per class).
