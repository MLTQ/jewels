# fit_corpus.py

## Purpose
The stage-1 -> stage-2 bridge: turn a directory of fixed-camera videos into a corpus of
per-window primitive-set checkpoints, unattended and resumable. These checkpoints are the
training set for the generative prior.

## Components

### `main()`
- **Does**: glob videos -> enumerate non-overlapping windows -> fit each -> checkpoint each;
  appends metrics to `corpus_log.jsonl`
- **Interacts with**: `data/video_io.py` (`count_frames`, `load_video(start_frame=...)`),
  `fit/fitter.py`

## Decisions
- **Resumable by checkpoint existence**, not by a state file — the filesystem is the state.
  A killed overnight run continues with no bookkeeping to corrupt.
- **Windowing at decode time** (`start_frame` sequential skip) rather than pre-splitting with
  ffmpeg: no re-encode artifacts, no intermediate files, one less tool in the loop. O(start)
  decode per window is noise next to a 3-minute fit.
- Windows are non-overlapping. Overlapping windows would leak near-duplicate sets between a
  future train/val split.
- Checkpoints carry a `source` key (video path + start frame) so a set can always be traced
  back to its pixels.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| stage 2 (future) | `<out>/<stem>_w<start>.pt` with keys state/cfg/info/source | Checkpoint schema — this plus fitter's info IS the frozen corpus format |

## Notes
- Defaults match the real-footage budget validated on the amplify clip (64f @ 160px short
  side, 3000→10000 prims, 3000 steps ≈ 3 min/window on the 4090).
- `--limit N` exists for smoke-testing a corpus before committing a night of GPU to it.
