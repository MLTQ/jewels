# `audit_coherent_source_oracle.py`

## Purpose

Runs Gate 2a8, the single-source coherence upper bound that decides whether independent block
selection is the cause of the hierarchy's qualitative texture failure.

## Components

### `main`

- **Does**: Fits the train-owned coherent realizer, scores correct/shuffled/null conditions on the
  immutable direct and source-disjoint sets, and renders the matched qualitative suite.
- **Does**: Records the selected training-source row for every generated window and labels the
  experiment as retrieval rather than prompt inference.
- **Rationale**: A recognizable source-disjoint row proves that the finite active Jewel vocabulary
  survives coherent generation; it does not prove novelty.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a8 protocol | One selected source program owns the entire window | Sampling ownership |
| Source-disjoint audit | Validation Jewel rows are never eligible for generation | Split semantics |
| Qualitative review | Correct, shuffled-scene, shuffled-block, and null rows share render settings | Arm parity |
