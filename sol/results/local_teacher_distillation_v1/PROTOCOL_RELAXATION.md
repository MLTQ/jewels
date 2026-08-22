# Post-distillation relaxation v1 — frozen protocol

## Question

The registered full-local arm improved exact LPIPS on all five styles but lost `0.1255 dB` exact
PSNR. Does equal render/structure refinement after removing the local targets preserve that
perceptual shift while recovering PSNR relative to an equally continued control?

## Source and matched controls

- Distilled source: `screens/full_seed0_600/encoder.pt` (`18.0399 dB`, `0.75963` exact LPIPS).
- Control source: `screens/control_seed0_600/encoder.pt` (`18.1654 dB`, `0.78204` exact LPIPS).
- Both arms receive 600 additional steps, seed 0, AdamW peak LR `1e-4`, warmup 100, and the same
  data, five held-out styles, sampling, renderer, teacher probability, and validation cadence as
  the registered local-teacher screen.
- Both arms use the unchanged global losses: Chamfer `0.2`, spread `0.02`, absolute size `0.023`
  with log offset `+0.35`, orientation `0.1`, direct tilt `1.0`, density `2.0`, sparsity `0.1`,
  polarization `0.01`, and target active fraction `0.58`.
- All local scale, axis, opacity, color, and color-gradient weights are exactly zero in both arms.
  This is a relaxation/fine-tuning test, not additional teacher-attribute exposure.

## Decision rule

The distilled-relaxation arm remains eligible only if sampled held-out occupancy is `<=0.985`,
active fraction is `<=0.70`, mixed tilt is `>=0.25`, and PSNR is no more than `0.50 dB` below the
matched relaxed control. Every eligible arm is exact-audited.

The relaxation hypothesis passes only if the distilled arm beats the matched control in both exact
PSNR and LPIPS while retaining the three structural gates. Promotion still requires exact PSNR
`>=20 dB` and LPIPS `<=0.40`. A failure rejects only this 600-step post-distillation schedule from
the registered full-local checkpoint; it does not erase the measured five-style LPIPS improvement
or generalize to higher capacity, other matching rules, or longer schedules.
