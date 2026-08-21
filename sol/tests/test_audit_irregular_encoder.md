# `test_audit_irregular_encoder.py`

## Purpose

Protects the active-count, center-layout, and mixed-spacetime structure semantics used by the
decisive irregular encoder audit.

## Components

### `IrregularAuditTests`
- **Does**: Verifies the layout selector applies opacity and coordinate-plane filters, then bounds
  plot size deterministically.
- **Does**: Verifies canonical 2% opacity filtering produces the expected active fraction and a
  bounded mixed-tilt statistic.
- **Does**: Verifies the preregistered three-seed gate passes qualifying metrics and fails when any
  seed retains excessive occupancy uniformity.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Irregular-field report | Active fraction and mixed tilt share canonical feature semantics | Metric definitions |
