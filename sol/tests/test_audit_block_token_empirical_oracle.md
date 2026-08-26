# `test_audit_block_token_empirical_oracle.py`

## Purpose

Checks that the empirical oracle audit assigns the correct, shuffled, and null programs to their
named arms before expensive rendering.

## Coverage

- Program ownership survives per-record likelihood aggregation.
- All three controls produce distinct expected macro values.
- Device lookup supports both reservoir and constellation realizer schemas.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a2 report | Correctly named program controls | Arm ownership changes |
