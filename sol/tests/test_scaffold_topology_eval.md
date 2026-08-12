# `test_scaffold_topology_eval.py`

## Purpose

Protects topology metric definitions, dense-count expansion, train-only threshold calibration, and
leakage-safe cross-class scaffold controls.

## Components

### `ScaffoldTopologyEvalTests`

- **Does**: Verifies perfect-count metric identities, valid threshold selection, class rotation,
  stable control names, the train-mean baseline, and canonical nested ranks with hard capacity
  rejection.
- **Interacts with**: `scaffold_topology_eval.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Research report | Exact counts score unit occupancy and slot F1 | Metric definitions |
| Held-out gate | Shuffled controls use another class at the same stride | Rotation policy |
| Checkpoint evaluation | Control keys remain stable | JSON schema |
| Frozen mark flow | Count expansion emits zero-based contiguous ranks per cell | Rank convention |
