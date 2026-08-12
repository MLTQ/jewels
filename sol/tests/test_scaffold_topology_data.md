# `test_scaffold_topology_data.py`

## Purpose

Protects the initial-plus-continuation topology dataset and the immutable carried-state raster used
by scaffold-conditioned density prediction.

## Components

### `ScaffoldTopologyDataTests`

- **Does**: Verifies frontier-zero inclusion, exact carry/birth partitioning, canonical count sums,
  and finite bounded carry channels.
- **Interacts with**: `scaffold_topology_data.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Initial generation gate | The first full stride is present with zero carry | Window filtering |
| Persistent rollout | Carried plus born IDs exactly partition active state | Ownership semantics |
| Topology encoder | Carry rasters have three finite channels | Raster schema |
