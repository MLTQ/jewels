# `audit_splat_density.py`

## Purpose

Audits fitted corpora for contribution-aware splat density and records a reproducible JSON report.

## Components

### `main`

- **Does**: loads one or more fitted corpora, measures every frame, aggregates raw frame vectors,
  and optionally writes the report
- **Interacts with**: `corpus.py` and `splat_density.py`
- **Rationale**: corpus-wide aggregation prevents one unusually broad or opaque window from defining
  the fit budget

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Density reports | Threshold keys and all summary statistics are explicit in JSON | Output schema |
| Fit-budget decisions | `--corpus` may be repeated and `--limit` is global | CLI semantics |
