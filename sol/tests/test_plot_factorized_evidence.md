# `test_plot_factorized_evidence.py`

## Purpose

Protects the v3 evidence figure from mixing minibatch training measurements with held-out evidence
or silently mislabeling the registered absolute-size arms.

## Components

### `FactorizedEvidencePlotTests`
- **Does**: Checks summary extraction, held-out-only progression parsing with compute offsets, and
  the size-run directory naming contract.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Evidence figure | Tests reflect trainer summary and JSONL schemas | Log schema |
| Size labels | Three/four-digit tokens retain two/three decimal weight precision | Naming convention |
