# `saliency_metrics.py`

## Purpose

Reports the moving-subject and stability errors that global PSNR/SSIM can hide, using only the
evaluation target video and its fitted reference background rather than semantic labels.

## Components

### `saliency_render_signature`

- **Foreground region**: Selects the top 20% of target pixels by background deviation, chroma, and
  spatial edge response, then reports RGB MAE and PSNR.
- **Foreground edge error**: Reports spatial-gradient MAE inside the selected region alongside its
  color metrics so sharper noise cannot masquerade as recovered structure.
- **Motion boundary**: Selects the top 20% of target temporal differences after adding their
  spatial-boundary response and reports temporal-gradient MAE there.
- **Quiet stability**: Reports temporal-gradient MAE in the quietest 50% of target changes, exposing
  flicker that a motion-weighted score would ignore.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Autonomous rollout report | Candidate and target share `(T,H,W,3)` and at least two frames | Video layout |
| Matched realizer ablation | Masks are target-owned and identical across candidate branches | Selection policy |
| Scientific record | Fractions remain 20% foreground, 20% motion, and 50% quiet by default | Metric meaning |

## Notes

- These are label-free diagnostic regions, not human segmentation masks.
- The metrics complement rather than replace global PSNR, SSIM, edge ratio, and temporal-change
  ratio; a branch must improve structure without merely amplifying gradients.
