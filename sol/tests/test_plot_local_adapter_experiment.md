# `test_plot_local_adapter_experiment.py`

## Purpose

Protects semantic mapping from generic audit seed labels to the declared local-adapter arms.

## Components

### `LocalAdapterPlotTests`
- **Does**: Builds a minimal synthetic result tree and verifies source, causal-control, and sampled
  diagnostic fields are collected under their correct semantic labels.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `plot_local_adapter_experiment.py` | Evidence mapping remains deterministic and provenance-aware | Result path or label changes |
