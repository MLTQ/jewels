# `test_fit_fine_block_language.py`

## Purpose

Prevents silent post-result changes to the Gate 2a6 routing shape or vocabulary capacity.

## Coverage

- The registered 16x16x8/K=1024 configuration is accepted.
- Coarser routing or a post hoc capacity change is rejected.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a6 protocol | Immutable fine-language settings | Validation changes |
