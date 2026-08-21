# `plot_factorized_evidence.py`

## Purpose

Turns the compact factorized-v3 experiment records into a pitch-readable four-panel figure. The
figure separates proposal capacity, structural-gate behavior, exact perceptual fidelity, and the
absolute jewel-size tradeoff so a structural success is not mistaken for an image-quality success.

## Components

### `summary_metrics`
- **Does**: Reads held-out trainer metrics and structure measurements from one summary.
- **Does not**: Use minibatch PSNR as evaluation evidence.

### `capacity_rows`
- **Does**: Requires exactly the three registered 10/20/36-slot capacity arms and sorts them by
  actual proposals per window.

### `progression_rows`
- **Does**: Extracts only held-out evaluation records and applies the source-checkpoint step offset.

### `size_weight` and `size_rows`
- **Does**: Decodes the registered directory naming convention and loads matched size arms in
  ascending loss-weight order.

### `main`
- **Does**: Plots capacity versus quality, all three structure gates, exact PSNR, and the registered
  size/quality Pareto region. The adjusted teacher extent is derived from the measured teacher
  extent and the declared log-space offset.
- **Does**: Preserves three-decimal labels for boundary-confirmation weights and marks points green
  only when they occupy the fixed size/PSNR acceptance region.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Evidence plot | Trainer summaries retain `latest_evaluation` and `jewels_per_window` | Summary schema |
| Size bracket | Run directories use `control_*`, `sizeNNN_*`, or `sizeNNNN_*` | Naming convention |
| Exact comparison | Audit retains `perceptual_macro` arms | Audit schema |
| Decision region | Thresholds match `SIZE_PROTOCOL.md` | Protocol revision |
