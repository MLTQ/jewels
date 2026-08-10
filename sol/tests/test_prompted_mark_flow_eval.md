# `test_prompted_mark_flow_eval.py`

## Purpose

Protects the leakage-safe stochastic prompt-control schema for oracle-topology mark generation.

## Components

### `PromptedMarkFlowEvaluationTests`
- **Does**: Builds a tiny multi-class corpus and verifies finite correct/shuffled/null fixed-path
  metrics for both full-prefix and text-only conditions.
- **Interacts with**: `prompted_mark_flow_eval.py`, `birth_mark_flow.py`, and `streaming_corpus.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Prompted mark-flow gate | Both context families expose all three text controls | Report schema |
