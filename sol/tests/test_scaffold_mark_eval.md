# `test_scaffold_mark_eval.py`

## Purpose

Protects the fixed-path scaffold-conditioning report used while training the initial-compatible
mark flow.

## Components

### `ScaffoldMarkEvalTests`

- **Does**: Builds a tiny two-class corpus and checks that finite aggregate, initial, and
  continuation control sections are produced. A paired single-class corpus verifies that only
  the scientifically unavailable shuffled control is omitted while null/context controls remain.
- **Interacts with**: `scaffold_mark_eval.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Training checkpoint | Evaluation report contains both temporal regimes | Section names |
| Scientific control | Every reported loss is finite under variable birth cardinality | Objective path |
