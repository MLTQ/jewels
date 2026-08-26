# `test_audit_jewel_casting_language.py`

## Purpose

Protects the Gate-0 audit statistics used to decide whether generator work is licensed. Synthetic
tests cover bundle decision counts, independent source-versus-serialized Jewel counts, the exact
grid-locking control, and same-versus-different program similarity aggregation.

## Components

### `JewelCastingAuditTests`
- **Does**: Fits a small vocabulary and verifies audit metrics, including the no-drop invariant,
  without requiring a renderer or GPU.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate-0 audit | Same-source and different-source pair groups are both mandatory | Pair aggregation |
| Grid control | Cell-center replacement reports complete locking | Irregularity definition |
