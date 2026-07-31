# video_io.py

## Purpose
Decode a clip to (T,H,W,3) float in [0,1], plus the synthetic falsification volume.

## Components

### `load_video(path, max_frames, resize, device)`
- **Does**: decode + optional area-resize
- **Rationale**: tries PyAV, then OpenCV, then imageio. One of these is always present on a
  training box and none is worth a hard dependency. Resize happens during load so peak host
  memory stays bounded on long clips.
- `resize` as an int means "short side, aspect preserved" and is what CLIs pass — the source
  aspect isn't known before decode, so callers can't compute (H, W) themselves. The tuple form
  remains for explicit control (and will distort if it disagrees with the source aspect).

### `synthetic_tube(T, H, W)`
- **Does**: a blob translating linearly — literally a sheared tube in (u,v,t)
- **Rationale**: THE falsification test. If a handful of anisotropic primitives can't nail this,
  the premise that motion lives in primitive orientation is wrong, and you find out in seconds
  instead of after a training run.

## Decisions
- Returns (T,H,W,3) channel-last to match the grid ordering in `core/volume.py` without a permute.
- `max_frames` defaults to 32, not None — a full-length clip will silently OOM otherwise.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `cli/fit_video.py`, `experiments/canonicalization.py` | (T,H,W,3) float in [0,1] | Layout or range |

## Notes
- Fixed camera, no cuts, for early experiments. See PROJECT.md sharp edges.
