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
- **Two-level recovery**: completed windows are skipped by final-checkpoint existence; an active
  window is atomically snapshotted to `<stem>_w<start>.recovery.pt` every 100 optimizer steps by
  default. A killed run therefore resumes within the window, including optimizer moments,
  densification tracker, history, background, and RNG position.
- Recovery files carry the full fit config and source identity. A mismatch fails loudly rather
  than combining optimizer state with different hyperparameters or pixels.
- Final corpus checkpoints are also written atomically. Only after that succeeds is the exact
  corresponding recovery file removed.
- **Windowing at decode time** (`start_frame` sequential skip) rather than pre-splitting with
  ffmpeg: no re-encode artifacts, no intermediate files, one less tool in the loop. O(start)
  decode per window is noise next to a 3-minute fit.
- Windows are non-overlapping. Overlapping windows would leak near-duplicate sets between a
  future train/val split.
- Checkpoints carry a `source` key (video path + start frame) so a set can always be traced
  back to its pixels.
- `--split-mode spatial` opts into temporal-preserving spatial densification. The mode is stored in
  `FitConfig`, so recovery rejects attempts to resume under a different split policy.
- Renderer culling and tiled-index controls are also stored in `FitConfig`. Support-mode corpus jobs
  fail loudly if `--support-capacity` cannot cover the conservative five-sigma spheres; they never
  silently fall back to KNN or truncate the candidate set. `support_tiled` is the scalable candidate
  under test; `support` remains its all-center oracle.
- `--geometry-constraint` is recorded and recovery-checked. Non-`free` modes are ablations and
  should use separate corpus output directories.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| stage 2 (future) | `<out>/<stem>_w<start>.pt` with keys state/cfg/info/source | Final checkpoint schema — this plus fitter's info IS the frozen corpus format |
| corpus recovery | `<out>/<stem>_w<start>.recovery.pt` with schema/cfg/source/fit_state | Recovery envelope or fitter recovery-state schema |

## Notes
- Defaults match the real-footage budget validated on the amplify clip (64f @ 160px short
  side, 3000→10000 prims, 3000 steps ≈ 3 min/window on the 4090).
- `--limit N` exists for smoke-testing a corpus before committing a night of GPU to it.
- `--recovery-every N` controls in-window durability; `0` disables it. At the validated 45k fit
  rate, the default 100-step interval limits typical lost GPU work to roughly two minutes.
- **[2026-08-04 process probe]** The CUDA CLI was terminated after an atomic step checkpoint,
  restarted against that file, and run to step 220. Its final field tensors, history, and background
  were bit-identical to an uninterrupted control; the recovery file disappeared only after final
  checkpoint replacement succeeded.
