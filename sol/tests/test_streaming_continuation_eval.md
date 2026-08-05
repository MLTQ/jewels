# `test_streaming_continuation_eval.py`

## Purpose

Protects the correct/shuffled/null continuation evaluation and exact carried-state merge.

## Components

### `StreamingContinuationEvaluationTests`

- **Does**: verifies finite evaluation metrics, zero carried-jewel error, and disjoint shuffled
  context/target intervals on a synthetic field
- **Interacts with**: `streaming_continuation_eval.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Continuation gate | Every metric is finite and carried state is copied exactly | Evaluation schema |
