# `test_plot_local_adapter_convergence.py`

## Purpose

Protects the compute-curve provenance and cumulative-step mapping for the frozen local appearance
adapter experiment.

## Components

### `LocalAdapterConvergencePlotTests`
- **Does**: Builds synthetic screen, training-log, continuation, and exact-audit artifacts.
- **Does**: Verifies the 4k continuation is plotted at 16k and derivative milestones retain their
  semantic arm label.
- **Does**: Verifies nested fixed-validation records are collected separately from training rows.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `plot_local_adapter_convergence.py` | Result paths and exact-audit positional mappings remain explicit | Path or mapping changes |
