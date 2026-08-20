# `test_temporal_tilt_ablation.py`

## Purpose

Protects the predeclared causal gate for the temporal-tilt ablation.

## Components

### `TemporalTiltAblationTests`
- Confirms the summary compares matched free and axis-aligned arms.
- Confirms the PSNR, free mixed-tilt, and projected-control checks all contribute to the gate.
- Confirms multiple seeds remain separate paired observations with Student-t uncertainty.
- Confirms a single observation does not claim a confidence interval.
- Confirms local reconstruction error is zero for an exact render.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `sol/temporal_tilt_ablation.py` | Largest matched seed pairs determine the causal gate | Summary or uncertainty semantics |
