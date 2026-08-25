# `plot_local_adapter_experiment.py`

## Purpose

Builds the semantic evidence figure for the frozen-base native local appearance experiment. It
maps generic audit seed labels back to the declared causal arms and keeps exact metrics separate
from sampled training diagnostics.

## Components

### `collect_evidence`
- **Does**: Loads the shared exact audits and final train-log records for source, radius controls,
  LPIPS strengths, and derivative variants.
- **Rationale**: `audit_irregular_encoder.py` labels candidate columns as seeds even when they are
  causal arms; an explicit mapping prevents a presentation label from changing provenance.

### `plot_evidence`
- **Does**: Draws the PSNR/LPIPS frontier, radius-2-minus-radius-0 causal deltas, objective/feature
  progression, and range/Jacobian diagnostics in one figure.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Local-adapter report | Audit directory names and candidate ordering match the declared protocol | Result-path or candidate-order changes |
| Scientific review | Exact metrics come from shared seven-frame audits; diagnostics are labeled sampled | Mixing metric provenance |
