# Perceptual (LPIPS) arm evaluation at native 288x192

The guide-upsample baseline showed reference-based PSNR/SSIM reward blur, so the same arms are
now scored where detail restoration counts: per-frame AlexNet LPIPS against the LTX target over
the identical 48-frame slice (`sol/perceptual_eval.py`, protocol in `report.json`). All four
held-out evaluation scaffolds; generated fields are the deterministic seed-31 native rollouts
(frozen v1 base re-rolled with `--correct-only`; coupled step-2,250 from the committed
`coupled_set_v1` run).

## Macro over four classes

| Arm | LPIPS (lower better) | PSNR | SSIM |
|---|---:|---:|---:|
| Fitted jewel ceiling | **0.0982** | 27.521 | 0.9831 |
| Guide upsample baseline | 0.6761 | 19.347 | 0.8596 |
| Generated coupled step2250 | 0.6911 | 14.475 | 0.6031 |
| Generated v1 base | 0.6971 | 14.142 | 0.6001 |

Per-class LPIPS, baseline vs coupled: Basketball 0.7834 vs **0.7274**; HorseRiding 0.7924 vs
**0.7027**; PlayingGuitar **0.5893** vs 0.6836; ApplyEyeMakeup **0.5394** vs 0.6508.

## Reading

1. **The representation is not the perceptual bottleneck.** The fitted 72k field scores 0.098
   LPIPS — 7x better than every other arm — so the ceiling for this representation is
   near-photoreal perceptual quality. The whole gap lives in generative realization, which is a
   training problem, not a representation problem.
2. **The perceptual ranking splits by content, unlike the pixel ranking.** On the two
   motion-heavy outdoor classes the generated fields beat the blur baseline (detail restoration
   is perceptually real); on the two stable close-up classes generated flicker costs more than
   blur. PSNR preferred the baseline in all four classes; LPIPS in only two — the metrics
   genuinely disagree, and the failure LPIPS localizes (temporal noise on smooth content) is the
   same appearance/stability bottleneck every recent gate has selected.
3. **The coupled birth-set direction is perceptually confirmed**: it improves LPIPS over the
   frozen base in three of four classes and on macro, independently of the feature-loss screen
   that selected it.
4. Macro LPIPS still favors the baseline by 0.015, so no perceptual victory is claimed yet. The
   scaling curve (`../data_scaling_v1/`) measures whether training data closes this gap.

Note the baseline's PSNR here (19.347 at 288x192) differs from the 48x80 report (21.081): the
bound depends on scoring resolution; each report states its own protocol.
