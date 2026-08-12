# `test_scaffold_topology_realizer.py`

## Purpose

Protects the explicit compatibility boundary between the 1,024-rank topology head and the frozen
512-rank continuation mark flow.

## Components

### `ScaffoldTopologyRealizerTests`

- **Does**: Verifies canonical expansion of valid learned counts and hard rejection of grid-shape
  mismatches or per-cell capacity overflow.
- **Interacts with**: `scaffold_topology_realizer.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Coupled renderer | Valid counts preserve their totals and canonical ranks | Expansion policy |
| Research gate | No predicted birth is silently clipped to the old 512-rank limit | Capacity policy |
