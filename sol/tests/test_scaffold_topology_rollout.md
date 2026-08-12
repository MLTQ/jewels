# `test_scaffold_topology_rollout.py`

## Purpose

Protects sequential oracle-mark materialization, rank-prefix matching, stable ID uniqueness, exact
carry, and contribution-aware density accounting.

## Components

### `ScaffoldTopologyRolloutTests`

- **Does**: Uses an exact-count stub to prove a perfect topology rollout recovers all target ranks
  and density with zero carry error, then verifies per-cell rank truncation.
- **Interacts with**: `scaffold_topology_rollout.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Sequential gate | Exact counts imply unit slot recall and density ratio | Oracle policy |
| Stable state | Previously emitted feature rows remain bit-identical | Merge behavior |
| Mark coupling | Predicted count selects a canonical rank prefix | Rank semantics |
