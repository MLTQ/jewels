# fitter.py

## Purpose
Per-clip stochastic-voxel fitting loop. This is stage 1 — the same stage GSVC/VeGaS occupy. It
exists to produce training data for the amortized/generative stage, so it's tuned for throughput
and reproducibility over last-dB quality.

## Components

### `FitConfig`
- **Does**: all hyperparameters for a fit

### `fit_volume(video, cfg, device)`
- **Does**: (T,H,W,3) in [0,1] -> (PrimitiveField, info dict)
- **Interacts with**: `make_grid`, `sample_indices`, `render_points`, `adapt`

## Decisions
- Per-parameter-group learning rates: geometry (mu/scale/quat) at half the appearance LR.
  Geometry moving as fast as color destabilizes early fitting.
- MSE only. Rejected adding SSIM/gradient loss for now — extra hyperparameters obscure the
  representation question.
- PSNR reported is computed on the sampled voxel batch, so it's noisy and NOT comparable to
  published full-frame PSNR. It's a training signal, not a benchmark number.
- Optimizer rebuilt after every adapt (see `adapt.md`).
- **[2026-07-31]** The voronoi mode and its steelman knobs (tau anneal, bg pseudo-cell, Lloyd)
  were removed with the branch — see PROJECT.md decision log. `FitConfig` no longer has a
  `mode` field; checkpoints made before this date carry one.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `cli/fit_video.py`, `cli/render_recon.py` | `(field, info)` with `info["history"]`, `info["background"]`, `info["shape"]` | info schema — freeze before corpus generation |
| `experiments/canonicalization.py` | `FitConfig(seed=...)` fully determines the run | Seeding behaviour |

## Notes
- The learned background is optimized jointly but lives outside `PrimitiveField`, so
  `fit_volume` returns it in `info["background"]` (a 3-float list — info is JSON-serialized by
  the CLIs). Anything reconstructing from a saved fit MUST add it back; `state_dict()` alone is
  not the full model.
