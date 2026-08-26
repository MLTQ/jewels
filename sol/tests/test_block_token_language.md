# `test_block_token_language.py`

## Purpose

Protects the Gate 2a block descriptor, discrete assignment, serialization, and continuous-coordinate
contracts.

## Coverage

- Local coordinates remain continuous and bounded.
- Every field produces one 77D descriptor and one token per routing block.
- The fitted vocabulary is finite and assignments are in range.
- Time-major Morton serialization is a complete permutation.
- The null-token selection is deterministic.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a protocol | Fixed descriptor and ordering semantics | Dimension or order changes |
