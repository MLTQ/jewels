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

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| stage 2 (future) | `runs/<name>/fit_seed<k>.pt` with keys state/cfg/info | Checkpoint schema — freeze this before generating a corpus |

## Notes
- `--size` is the short side; aspect is preserved from the source (handled inside
  `load_video`).
- Checkpoints saved before 2026-07-31 were named `<mode>_seed<k>.pt` and carry a `mode` key in
  cfg (the removed voronoi-vs-additive A/B era).
