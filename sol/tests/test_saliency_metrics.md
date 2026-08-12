# `test_saliency_metrics.py`

## Purpose

Protects the label-free foreground, motion-boundary, and temporal-stability evaluation contract.

## Components

### `SaliencyMetricTests`

- **Does**: Verifies exact videos score zero/100 dB, a missing moving saturated pixel produces
  foreground/edge/motion error, and candidate-only flicker appears in the target-quiet metric.
- **Interacts with**: `saliency_metrics.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Realizer ablation | Exact candidates define the zero-error floor | Metric baseline |
| Stability gate | Target-static candidate changes cannot disappear from the report | Quiet mask |
