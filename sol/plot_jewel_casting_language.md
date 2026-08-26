# `plot_jewel_casting_language.py`

## Purpose

Turns the registered casting-language JSON audit into one presentation-ready evidence figure. The
panels expose both supporting and adverse results rather than reducing the experiment to a verdict.

## Components

### `plot_payload`
- **Does**: Validates the report schema and extracts vocabulary-ordered motif, rendering,
  canonicality, and lattice-control series.
- **Rationale**: Keeping extraction independent of plotting makes report interpretation testable.

### `main`
- **Does**: Draws four panels with the preregistered thresholds and the final gate verdict.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Pitch evidence | Gate-v0 report macro and canonicality keys | Report schema or metric names |
| Scientific review | Same/different controls and grid negative control remain visible | Removing control series |
