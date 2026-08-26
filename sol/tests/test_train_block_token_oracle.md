# `test_train_block_token_oracle.py`

## Purpose

Protects the Gate 2a control ownership and batch semantics independently of the expensive GPU run.

## Coverage

- Cyclic shuffling changes prompt source while preserving within-source fit rank.
- Oracle, shuffled, and null arms report density plus all three active Jewel roles.
- Misaligned record/program collections fail explicitly.
- A prior report can own the full source split without duplicated command-line lists.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a report | Matched, correctly owned control programs | Shuffle or batch changes |
