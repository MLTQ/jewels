# `test_scaffold_topology.py`

## Purpose

Protects scaffold-topology tensor shapes, decomposed objectives, capacity-bounded decoding, and
zero-carry initial generation.

## Components

### `ScaffoldTopologyTests`

- **Does**: Exercises batched/unbatched forward and loss paths, implicit zero carry, decoded count
  bounds, and invalid shape/threshold rejection.
- **Interacts with**: `scaffold_topology.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Trainer | Batched losses return four named finite terms | Objective schema |
| Initial window | Missing carry is exactly equivalent to an all-zero carry raster | Null-carry semantics |
| Mark realizer | Decoded counts are non-negative and capacity bounded | Decode policy |
