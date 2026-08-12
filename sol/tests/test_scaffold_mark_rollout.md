# `test_scaffold_mark_rollout.py`

## Purpose

Protects the empty-state initial generation and two subsequent strides of append-only model state.

## Components

### `ScaffoldMarkRolloutTests`

- **Does**: Uses tiny deterministic topology and stochastic mark models to verify three complete
  strides, all-rank synthesis, contiguous stable IDs, and zero prior/carry mutation.
- **Boundary control**: Asserts the default policy is censored only at frontier zero and that the
  matched legacy control can make that first frontier strict.
- **Interacts with**: `scaffold_mark_rollout.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Autonomous demo | Three guides create initial plus two continuations | Frontier sequence |
| Jewel editor | IDs are contiguous and existing features remain bit-identical | Append policy |
