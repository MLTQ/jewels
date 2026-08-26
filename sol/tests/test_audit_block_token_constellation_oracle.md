# `test_audit_block_token_constellation_oracle.py`

## Purpose

Checks the explicit count-adjustment evidence emitted by Gate 2a4.

## Coverage

- Mean unadjusted count, mean adjustment, and worst adjustment are computed from the named arm.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a4 report | Transparent exact-count normalization | Realization schema |
