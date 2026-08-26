# `test_aggregate_prompt_jewel_scaling.py`

## Purpose

Protects the frozen sign convention and pass logic for the prompt-to-Jewel data-scaling curve.

## Components

### `PromptScalingAggregationTests`

- **Does**: Verifies that positive NLL margin means the correct prompt is better, all four
  monotonic checks are required, and duplicate data sizes are rejected.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Scaling evidence | Margins and retrieval are calculated directly from gate reports | Metric math |
