# `test_aggregate_prompt_repetition.py`

## Purpose

Protects the exact-prompt repetition curve's sign convention and frozen monotonic pass logic.

## Components

### `PromptRepetitionAggregationTests`

- **Does**: Verifies density/token margins, retrieval math, final gate propagation, and that every
  registered scaling family is required.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1e evidence | Control-minus-correct margins and nondecreasing curve checks | Metric math |
