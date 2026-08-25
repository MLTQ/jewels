# Local appearance adapter v1 protocol

## Source ownership

- Source: `frozen_appearance_v1/continuation/frozen_render_total1200/encoder.pt`.
- Five held-out cooking clips, one per style; seven shared audit frames.
- Geometry and all source appearance/background tensors frozen bitwise.
- Only zero-expanded adapter tensors train.

## Declared stages

1. Radius-0 versus radius-2 raw-neighborhood render controls, 400 updates.
2. Radius-0 versus radius-2 LPIPS 0.01 control and radius-2 LPIPS 0.05 strength screen.
3. Bias-free derivative-only adapter at scales 1 and 32; scale 32 selected from the measured
   pre-outcome gradient ratio, not audit performance.
4. Shared exact PSNR/LPIPS/SSIM/layout audit and visual sheet; replicate only an arm below LPIPS 0.70.

The beads record `jewels-h0s.6` preserves the chronological preregistration and outcome notes.

## Interpretation policy

Successful metric movements are mechanism evidence. Failed arms reject only their declared input,
capacity, loss scale, and duration. No single screen is treated as a law against local evidence or
Gaussian video representations.
