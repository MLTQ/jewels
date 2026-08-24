# Frozen render continuation v1 — registered protocol

## Trigger and arm selection

The registered seed-0 audit retained frozen-render as the unique metric winner. Relative to the
frozen source it improved exact PSNR `19.66478 -> 19.89236 dB`, LPIPS `0.73288 -> 0.72605`, SSIM
`0.82256 -> 0.83372`, and layout PSNR `23.04958 -> 23.49562`. It improved both PSNR and LPIPS in all
five styles and held every geometry prediction bitwise exact.

Frozen-perceptual reached `19.86420 / 0.72712` and lost both metrics to frozen-render in every style,
so it is not continued. Frozen-stabilized reached `19.85251 / 0.72836`; its explicit penalties
reduced sampled out-of-range RGB, residual energy, and temporal-change overshoot within the
registered `0.05 dB / 0.002 LPIPS` stability budget, but it cannot rescue a dominated perceptual
base objective. This supports range stabilization as a future control, not selection of this arm.

## Registered continuation

- Continue `screens/frozen_render_seed0_600/encoder.pt` for exactly 600 additional optimizer steps,
  reported as frozen-appearance total step 1,200.
- Preserve the selected objective exactly: sampled render MSE only; all appearance-grid, range,
  residual-energy, teacher-structure, local, responsibility, sparsity, and density weights remain
  zero.
- Preserve seed 0, the 120-video manifest, source selection, AdamW peak LR `1e-4`, warmup 100,
  1,024 sampled points, support-tiled capacity 8,192, and the same five held-out sources.
- Preserve `--freeze-geometry`; all held-out center, scale, quaternion, and opacity tensors must
  remain bitwise equal to the original pre-continuation frozen source, not merely the step-600 arm.
  Since the step-600 arm already proved source equality, the transitive check is exact.
- As in earlier declared continuations, optimizer state and the 600-step cosine schedule restart for
  the new tranche. No wall-time comparison is model evidence.

## Decision rules

The continuation passes if exact macro PSNR reaches `>=20 dB`, exact LPIPS is no worse than
`0.72605`, both metrics improve over the original `19.66478 / 0.73288` source in every style, and
all geometry/structure gates remain exact. Contact sheets must not show a qualitative collapse or a
material increase in isolated dark speckles.

If it passes, replicate the complete two-tranche frozen-render path from the original source with
continuation seeds 1 and 2. Each seed must remain geometry-exact and cross exact 20 dB without an
LPIPS regression. This is appearance-optimization replication from one upstream geometry state;
independent upstream-encoder replication remains separate future work.
