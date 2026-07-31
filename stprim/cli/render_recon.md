# render_recon.py

## Purpose
Visual counterpart to the metric scripts: fit one clip, render the full volume, and emit
human-viewable artifacts. Metrics say how well the fit converged; this shows what the
representation actually looks like.

## Components

### `reconstruct(field, info, cfg, device)`
- **Does**: full-volume render -> (T,H,W,3) clamped to [0,1]
- **Rationale**: re-adds `info["background"]` (see fitter.md — the background is not in the
  field's `state_dict`).

### `heat(err)` / `to_pil` / `hstack` / `vstack`
- **Does**: minimal black->red->yellow error map and PIL montage helpers
- **Rationale**: PIL-only on purpose; the training box has no matplotlib/torchvision/imageio
  (see data/video_io.md for the same dependency-tolerance stance).

## Outputs
- `compare.gif` — [GT | recon] per frame, looped, nearest-upscaled
- `contact_sheet.png` — sampled frames x {GT, recon, err x5}
- `fit_seed{s}.pt` — checkpoint (state + cfg + info; info carries the background)

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| (none yet) | | |

Depends on `fit_volume` returning `info["background"]` and `info["shape"]`.

## Notes
- Defaults mirror `experiments/canonicalization.py`'s defaults so the rendered fit is the same
  experiment, one seed.
- Reported PSNR here is true full-volume PSNR — comparable across runs, unlike the noisy
  sampled-batch PSNR in the fit loop (see fitter.md).
