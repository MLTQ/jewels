# `test_audit_individual_jewel_language.py`

## Purpose

Protects the active three-role language definition and same-versus-different aggregation without
allowing the constant individual-layout coordinate back into the metric.

## Components

### `ActiveIndividualLanguageTests`
- **Does**: Freezes active roles and validates a positive repeated-field similarity margin.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 0f | Layout is absent; covariance/surface/gradient are active | Active-factor tuple |
