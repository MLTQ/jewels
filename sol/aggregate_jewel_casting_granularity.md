# `aggregate_jewel_casting_granularity.py`

## Purpose

Aggregates matched factorized-language audits at 8, 4, 2, and 1 Jewels per cast and evaluates the
registered one-Jewel-plus-centroid upper-bound gate. The result separates generative decision cost
from rendering and language stability; it is not a compression score.

## Components

### `evaluate_granularity_gate`
- **Does**: Validates report ownership, extracts the K=1,024 curve, computes equivalent eight-frame
  decisions, and evaluates the frozen bundle-1 and monotonicity checks.

### `main`
- **Does**: Saves a compact aggregate JSON report and a four-panel pitch-readable curve.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 0c protocol | Bundle sizes 8/4/2/1, 49-frame sources, four role tokens, K1024 rows | Protocol ownership |
| Pitch evidence | Energy, render, canonicality, and decision-cost panels retain thresholds | Aggregate schema |
