# `test_audit_factorized_jewel_casting_language.py`

## Purpose

Protects no-drop accounting, per-role energy reporting, and composite same-versus-different
canonicality statistics used by registered Gate 0b.

## Components

### `FactorizedCastingAuditTests`
- **Does**: Fits a small synthetic compositional language and validates report invariants without a
  renderer or GPU.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 0b | Four role metrics and composite canonicality margin | Report aggregation |
