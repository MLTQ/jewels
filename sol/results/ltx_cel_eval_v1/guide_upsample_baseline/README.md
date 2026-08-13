# Guide-upsample baseline: cel-shaded arm

Same protocol as `../../ltx_scaffold_v1/guide_upsample_baseline/` (see that README for the
construction), applied to the four cel-shaded style-reconstruction fields and joined against
the adapted-generator deterministic rollout report
(`../generator_adaptation/rollout_summary.json`).

## Results (48 frames, four same-field cel rollout targets)

| Arm | PSNR | SSIM | edge ratio | temporal-change ratio | fg PSNR | quiet MAE |
|---|---:|---:|---:|---:|---:|---:|
| Guide upsample (this baseline) | **19.604** | **0.6811** | 0.434 | 0.556 | **14.23** | **0.00203** |
| Generated correct (seed 31) | 15.401 | 0.4270 | **1.122** | 6.410 | 11.07 | 0.02973 |
| Fitted jewel ceiling | 23.131 | 0.8974 | 1.390 | 2.243 | — | — |

Per class, baseline vs generated-correct PSNR: Basketball 22.864 vs 17.294; HorseRiding
19.948 vs 16.578; PlayingGuitar 19.265 vs 14.521; ApplyEyeMakeup 16.340 vs 13.211.

## Reading

1. The trivial decode again wins every class on PSNR (+2.2 to +5.6 dB), global SSIM, and
   foreground PSNR, with 14.6x lower quiet-region temporal error.
2. The cel domain narrows the PSNR gap relative to photoreal (macro +4.2 dB here vs +6.5 dB)
   — flat regions upsample well but ink contours blur, mirroring the fill/contour split of
   the fixed-density fit gate.
3. The generated field's temporal-change ratio (6.41) is far *above* target here — the
   instability is generated flicker, not missing motion; the baseline's 0.556 is missing
   motion. Neither matches the target; they fail in opposite directions.
4. Basketball's low baseline SSIM (0.472) shows global SSIM behaving erratically on
   near-flat cel frames; conclusions should rest on the error metrics jointly, and on
   perceptual metrics once added.

Artifacts: `report.json`, `contact_sheet.png`.
