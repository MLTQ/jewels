# `perceptual_eval.py`

## Purpose

The guide-upsample baseline proved that reference-based PSNR/SSIM prefer a blurred decode of the
stack's own input over the generated field. This module scores the same arms where detail
restoration counts — per-frame LPIPS against the scaffold target — so the detail-energy claim is
tested instead of asserted. One protocol, all arms: trivial guide decode, fitted ceiling, and any
number of saved generated fields.

## Components

### `lpips_metric`
- **Does**: Builds a batched LPIPS callable (AlexNet by default) on the requested device.
- **Rationale**: `lpips` is imported lazily inside the factory so the core test suite and every
  other sol module keep zero perceptual-net dependencies.

### `score_arms`
- **Does**: For each named arm, records per-frame LPIPS, its mean, the standard
  `render_signature`, and a `layout_signature` on the identical target slice.
- **Interacts with**: `render_signature` in `audit_prompted_washout.py` so PSNR/SSIM stay
  cross-checkable against every rollout report.

### `layout_signature`
- **Does**: Average-pools texture away (factor 8 at native resolution, adaptive on small
  inputs) and scores PSNR/SSIM on the pooled videos — where things are, not how they are
  textured.
- **Rationale**: The domain-matched audit showed patch-based LPIPS rewarding local texture
  statistics while macro-layout visibly regressed; the battery needs a metric pointed at
  exactly the thing LPIPS misses.

### `main`
- **Does**: Loads validation sources via `load_prompted_fields`, rebuilds the guide-upsample
  baseline per stride, renders the fitted field (background re-read from the fitted checkpoint,
  as in the rollout renderer) and each `--field label=dir` generated-field checkpoint with its
  own stored background, then writes `report.json` with per-source and macro numbers.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Scientific report | LPIPS and PSNR/SSIM computed on one shared reference slice | Protocol drift |
| `sol/results/*/perceptual_eval*/` | Schema `perceptual-arm-eval-v1` | Report fields |
| Generated-field inputs | `<source_id>_generated_field.pt` with `features`/`background` | Rollout field schema |

## Notes

- Four validation clips are far too few for FVD; per-clip LPIPS is the honest perceptual step
  until a larger held-out corpus exists.
