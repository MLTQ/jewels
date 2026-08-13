# `guide_upsample_baseline.py`

## Purpose

Answers the reviewer question the rollout reports leave open: how much fidelity does the
learned topology+mark stack add over trivially decoding its own input? The generation stack's
only pixel input per stride is the `(16,16,8)` cell-RGB guide; this module upsamples exactly
those guides back to render resolution and scores them under the identical rollout protocol.

## Components

### `cell_raster_to_video`
- **Does**: Inverts `video_to_cell_raster`'s flatten/permute and trilinear-upsamples one
  `(cells,3)` guide to `(frames,H,W,3)`.
- **Interacts with**: `GridSpec` in `token_grid.py`; must mirror the canonical
  `(u*gv+v)*gt+t` cell order or the baseline silently scores a scrambled video.

### `guide_upsample_baseline`
- **Does**: Rebuilds the per-stride guides exactly as `render_scaffold_mark_rollout.py` does
  (from the video already resized to render resolution) and concatenates their decodes.
- **Rationale**: Piecewise-per-stride decoding (not one whole-clip resample) matches the
  information boundary the rollout enforces; its stride seams are part of the baseline.

### `evaluate_source`
- **Does**: Scores the decode with `render_signature`, `saliency_render_signature`, and the
  rollout's `_seam_report` against the same `video[:completed_frames]` reference.

### `main`
- **Does**: Walks a training manifest, joins each validation source against an existing
  rollout `summary.json` (reusing its per-source fitted background for saliency metrics and
  its pipeline arm signatures), writes `report.json` plus a target/baseline contact sheet.
- **Interacts with**: `--video-root` remaps remote manifest video paths by basename so the
  audit can run off-box on committed or copied MP4s.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Scientific report | Baseline and pipeline numbers share one protocol and reference | Metric or slicing drift |
| `sol/results/*/guide_upsample_baseline/` | `report.json` schema `guide-upsample-baseline-v1` | Schema fields |
| Paper baseline table | `macro_baseline_render_signature` beside `macro_render_signatures` | Aggregation change |

## Notes

- The baseline consumes only what the stack consumes (guide rasters); it never sees fitted
  jewels, carried state, or text. The fitted background enters saliency *metrics* only, as in
  the rollout report.
- Imports the rollout's private `_seam_report` deliberately: protocol identity outweighs the
  private-import smell, and the repo already cross-imports private panel helpers.
