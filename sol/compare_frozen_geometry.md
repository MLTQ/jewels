# `compare_frozen_geometry.py`

## Purpose

Compares the parameters that own factorized jewel geometry directly across checkpoints. This is a
stronger freeze proof than tolerance-based aggregate structure metrics or prediction sampling.

## Components

### `geometry_state`

- **Does**: Selects every `geometry_trunk.*` and `geometry_head.*` state tensor and rejects a
  checkpoint without the factorized geometry contract.

### `compare_geometry_states`

- **Does**: Requires identical tensor names/shapes, applies `torch.equal` to every tensor, and
  reports the exact mismatches and maximum absolute change.

### `main`

- **Does**: Compares any number of candidate checkpoints against one source on CPU and writes a
  versioned JSON report.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Frozen-appearance replication | Every geometry tensor remains bitwise source-identical | Prefix ownership |
| Scientific report | Candidate paths and mismatches remain explicit | Report schema |
