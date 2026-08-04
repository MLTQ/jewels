# `cache_motion_points.py`

## Purpose

Builds deterministic, label-free foreground coordinate pools from each fitted window's original
source video. The pools let tokenizer training target small moving figures directly.

## Components

### `_window_motion_points`
- **Does**: Scores pixels by distance from the temporal median plus adjacent-frame change, then keeps
  an equal top-k quota from every frame and converts `(u,v,t)` to the field's `[-1,1]` coordinates.
- **Rationale**: Per-frame quotas prevent one dramatic instant from consuming the pool and guarantee
  temporal coverage of an object's path.

### `main`
- **Does**: Reads source metadata from corpus checkpoints, decodes each source video once in order,
  resizes exactly to the fitted shape, and writes one fixed-size `.motion.pt` sidecar per window.
- **Interacts with**: `train_dense_autoencoder.py` consumes the point tensors; corpus checkpoints
  remain unchanged.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dense trainer | Sidecar contains `(P,3)` normalized `points` | Sidecar schema |
| Reproducibility | Coordinates use checkpoint source/start/shape and balanced frame quotas | Sampling policy |

## Notes

- PyAV decodes each source sequentially once and resizes frames before tensor conversion, keeping
  memory bounded to one 64-frame window.
- This is training-only information from the training videos, not held-out labels or evaluation data.
