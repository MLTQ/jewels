# `test_splat_density.py`

## Purpose

Protects the effective-density definition used to size future fitted-jewel budgets.

## Components

### `SplatDensityTests`

- **Does**: verifies temporal-plane support, peak-alpha thresholding, equal-weight participation
  ratio, summaries, and argument validation
- **Interacts with**: `splat_density.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Density methodology | Temporal marginal sigma and renderer sigmoid weight define peak alpha | Metric definition |
| Corpus audit | Count summaries remain JSON-safe | Summary schema |
