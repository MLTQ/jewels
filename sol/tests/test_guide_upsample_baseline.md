# `test_guide_upsample_baseline.py`

## Purpose

Guards the two failure modes that would silently invalidate the baseline: a scrambled axis
order in the raster inverse, and a reference slice that drifts from the rollout protocol.

## Components

### `test_constant_video_round_trips_exactly`
- **Does**: A constant video must survive guide encode/decode bit-near-exactly; any
  normalization or ordering bug breaks this immediately.

### `test_inverse_preserves_axis_order`
- **Does**: Lights one spatiotemporal corner (early frames, top rows, right columns) and
  asserts the decode's bright mass stays in that corner — pinning u=width, v=height, t=time
  through the `(u*gv+v)*gt+t` flatten.

### `test_rejects_incomplete_strides` / `test_evaluate_source_matches_rollout_reference_slice`
- **Does**: Stride bounds are validated, and the evaluation reports the rollout's
  `strides*stride_frames` reference slice with render/saliency/seam fields present.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `guide_upsample_baseline.py` | These invariants hold for every future edit | Axis order, slicing |
