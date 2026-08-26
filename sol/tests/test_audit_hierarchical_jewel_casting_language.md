# `test_audit_hierarchical_jewel_casting_language.py`

## Purpose

Protects hierarchical role composition, active decision accounting, exact full-residual round trip,
and source-similarity aggregation before fresh GPU validation.

## Components

### `HierarchicalCastingAuditTests`
- **Does**: Fits tiny pair/individual vocabularies and validates hierarchy invariants without a
  renderer or GPU.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 0d | Pair dimensions 0:9, individual dimensions 9:22, active 2+2 roles | Composition contract |
