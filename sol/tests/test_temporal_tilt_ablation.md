# `test_temporal_tilt_ablation.py`

## Purpose

Protects the predeclared causal gate for the temporal-tilt ablation.

## Components

### `TemporalTiltAblationTests`
- Confirms the summary compares matched free and axis-aligned arms.
- Confirms the PSNR, free mixed-tilt, and projected-control checks all contribute to the gate.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `sol/temporal_tilt_ablation.py` | Largest matched budget determines the causal gate | Summary semantics |
