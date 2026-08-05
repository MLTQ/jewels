# `render_streaming_continuation.py`

## Purpose

Renders the learned continuation against its fitted future under correct, disjoint-shuffled, and
null prefix controls so the numerical selectivity gate has an inspectable video artifact.

## Components

### `frame_points`

- **Does**: builds a low-resolution global `(u,v,t)` grid for one future commit interval

### `main`

- **Does**: restores the fitted field and continuation checkpoint, predicts identical oracle birth
  ranks under three prefix conditions, merges exact carried jewels, and writes a labeled GIF,
  contact sheet, and full-grid field PSNR report
- **Interacts with**: `streaming_data.py`, `streaming_model.py`,
  `streaming_continuation_eval.py`, and `render.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Visual gate | Every condition shares target birth ranks and render points | Target construction |
| Shuffled control | Prefix interval is disjoint from the target future stride | Control selection |
| Persistent state | Candidate fields concatenate unchanged carried jewels with predicted births | Merge semantics |

## Notes

- Rendering defaults to `24×32` and nearest-neighbor upscale because the all-jewel reference
  renderer is quadratic in pixels and active jewels. This is a structural diagnostic, not a
  presentation-quality reconstruction.
- `--fit-checkpoint` overrides the source path embedded in a relocated continuation checkpoint.
