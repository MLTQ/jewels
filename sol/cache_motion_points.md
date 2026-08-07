# `cache_motion_points.py`

## Purpose

Builds deterministic, label-free foreground coordinate pools from each fitted window's original
source video. The pools combine motion and saturated-color samples so tokenizer training targets
small moving figures and rare chroma instead of being dominated by static background.

## Components

### `_window_motion_points`
- **Does**: Preserves the historical motion-only helper as a compatibility wrapper.

### `_window_saliency_points`
- **Does**: Splits each frame's fixed quota between temporal-median/change scores and RGB chroma,
  removes duplicate selections, and converts `(u,v,t)` to the field's `[-1,1]` coordinates.
- **Rationale**: Per-frame quotas prevent one dramatic instant from consuming the pool and guarantee
  temporal coverage; disjoint motion/chroma quotas keep rare clothing colors in the training loss.

### `main`
- **Does**: Reads one or more corpus roots, decodes each source video once in order, resizes exactly
  to the fitted shape, and writes one fixed-size `.motion.pt` sidecar per window with sample kinds.
- **Interacts with**: `train_dense_autoencoder.py` consumes the point tensors; corpus checkpoints
  remain unchanged.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dense trainer | Sidecar contains `(P,3)` normalized `points`; added keys are optional | Point schema |
| Reproducibility | Coordinates use checkpoint source/start/shape and balanced disjoint quotas | Sampling policy |

## Notes

- PyAV decodes each source sequentially once and resizes frames before tensor conversion, keeping
  memory bounded to one 64-frame window.
- The default pool is 50% motion and 50% chroma; `--chroma-fraction 0` reproduces motion-only pools.
- This is training-only information from the training videos, not held-out labels or evaluation data.
