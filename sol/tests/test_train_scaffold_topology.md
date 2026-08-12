# `test_train_scaffold_topology.py`

## Purpose

Protects trainer-side whole-source split filtering, explicit diagnostic source overrides,
per-stride train-mean construction, and held-out control-view ownership.

## Components

### `TrainScaffoldTopologyTests`

- **Does**: Builds synthetic prepared sources and verifies that validation counts never enter the
  training mean while control rows preserve source/target identity. It also verifies that the
  diagnostic cross-validation override is source-exact and rejects duplicate, missing, or
  all-validation source sets.
- **Interacts with**: `train_scaffold_topology.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Leakage-safe evaluation | Train and validation flattening remain source-owned | Split logic |
| Baseline | Count priors average only training rows of the same stride index | Mean policy |
| Controls | Validation target counts remain attached to their original source | Ownership schema |
| Diagnostic split | Only explicitly named sources validate and the input objects are not mutated | Override semantics |
