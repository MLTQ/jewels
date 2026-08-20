# `audit_encoder_convergence.py`

## Purpose

Adds perceptual, layout, and covariance-structure evidence to the sampled-voxel PSNR convergence
curve, and produces qualitative frames suitable for inspecting whether scaling merely smooths or
actually preserves video content.

## Components

### `structure`
- Samples a deterministic 4,096-slot cross-section, filters inactive splats, converts the precision
  Cholesky to covariance, and reports anisotropy and mixed space/time tilt. Mixed tilt is zero for
  purely spatial or temporal axes and one for a balanced diagonal tube.

### `main`
- Audits one frozen validation clip per visual style across all curve points and all seeds for
  structure.
- On the first seed, support-renders seven evenly spaced full frames and reports LPIPS,
  PSNR/SSIM, and pooled layout metrics against the source scaffold.
- Writes raw records, macro summaries, a metric plot, and a target-versus-budget contact sheet.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Scientific report | Same validation identities, frames, renderer, and perceptual seed at every size | Audit protocol |
| Pitch evidence | Qualitative columns are target, n12, n60, n180 for every style | Contact sheet order |
