# `test_plot_irregular_evidence.py`

## Purpose

Protects the evidence plot from mixing minibatch training metrics with held-out evaluation metrics.

## Components

### `IrregularEvidencePlotTests`
- **Does**: Verifies non-evaluation JSONL rows are ignored and evaluation structure values retain
  their declared meanings.
- **Does**: Verifies matched-control points come from the held-out summary record.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Evidence figure | Only held-out evaluation records define progression curves | Parser behavior |
