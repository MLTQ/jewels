# `test_aggregate_jewel_casting_granularity.py`

## Purpose

Protects registered Gate-0c ownership, upper-bound checks, monotonic curve interpretation, and
eight-frame decision accounting.

## Components

### `CastingGranularityTests`
- **Does**: Evaluates a synthetic passing curve and rejects incomplete bundle-size coverage.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 0c | Bundle-1 primary arm, bundle-8 baseline, 49-frame decision conversion | Gate thresholds |
