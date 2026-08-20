# `fit_encoder_teacher_subset.py`

## Purpose

Refits a small, style-stratified set of direct optimization ceilings with the proven
support-complete renderer. These are trusted positive teachers for judging encoder structure and
fidelity; they do not supervise the current render-loss encoder.

## Components

### `main`
- Selects exactly one frozen validation clip per visual style.
- Fits up to 72,000 free, P1-color splats for 3,000 steps with five-sigma tiled support, using a
  declared 100-step/15% densification schedule, and saves full-provenance checkpoints.
- Support-renders seven fixed frames and reports PSNR/SSIM, LPIPS, anisotropy, and mixed spacetime
  tilt, with target-versus-teacher qualitative images.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Encoder audit | Teacher identities come from the same frozen validation manifest | Selection policy |
| Feasibility report | Every teacher records support renderer, sigma, seed, compute, and structure | Report schema |
