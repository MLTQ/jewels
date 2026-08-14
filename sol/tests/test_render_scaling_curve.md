# `test_render_scaling_curve.py`

## Purpose

Keeps the curve honest at the plumbing level: points carry the conditioning margin next to the
headline metrics, unknown report arms fail loudly, and the figure actually renders.

## Components

### `test_collect_point_extracts_metrics_and_margin` / `test_collect_point_rejects_unknown_arm`
- **Does**: Schema extraction and arm validation against synthetic summaries.

### `test_render_curve_writes_figure`
- **Does**: Three synthetic points render to a non-trivial PNG under the Agg backend.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `render_scaling_curve.py` | Input schemas stay `latest_evaluation.aggregate` + report `macro` | Schema drift |
