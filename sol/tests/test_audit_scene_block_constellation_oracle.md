# `test_audit_scene_block_constellation_oracle.py`

## Purpose

Protects the independent disruption of global scene syntax and local block syntax in Gate 2a5.

## Coverage

- Correct hierarchy preserves both levels.
- Shuffled-scene changes only the global token.
- Shuffled-block changes only the local program.
- Null hierarchy replaces both levels.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a5 report | Causally isolated hierarchy controls | Arm ownership |
