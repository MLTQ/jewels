# fit_video.py

## Purpose
Entry point. Fits one clip and writes a checkpoint + summary. These checkpoints are the
stage-2 training data.

## Components

### `main()`
- **Does**: arg parsing, load/synthesize, fit, save `.pt`, print final PSNR
- **Interacts with**: `data/video_io.py`, `fit/fitter.py`

## Decisions
- Saves full `state_dict` + cfg + info per run. `info["background"]` is part of the model —
  see fitter.md.
- Exposes `--cull-mode` and all finite-support budget controls. This keeps the fast historical KNN
  baseline available while allowing a checkpoint to record that it was trained with support-safe
  candidates. `--cull-mode exact` is intentionally limited to tiny audit cases.
- `--geometry-constraint axis_aligned|isotropic` exposes the fitter's causal controls. The default
  `free` is the only production representation.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| stage 2 (future) | `runs/<name>/fit_seed<k>.pt` with keys state/cfg/info | Checkpoint schema — freeze this before generating a corpus |

## Notes
- `--size` is the short side; aspect is preserved from the source (handled inside
  `load_video`).
- Checkpoints saved before 2026-07-31 were named `<mode>_seed<k>.pt` and carry a `mode` key in
  cfg (the removed voronoi-vs-additive A/B era).
