# `token_grid.py`

## Purpose

Defines raster cell indexing, occupancy statistics, and lossless canonical slot packing. It replaces
silent truncation with an explicit capacity contract and supplies spatial structure to the learned
autoencoder.

## Components

### `GridSpec`
- **Does**: Owns raster ordering, normalized coordinate bucketing, cell AABB queries, and slot budget.
- **Rationale**: Grid and capacity are checkpoint semantics, not incidental hyperparameters.

### `OccupancyGrid.pack`
- **Does**: Canonically sorts jewels within each cell and packs them into masked slots.
- **Rationale**: Raises `GridCapacityError` on overflow; dropping a target is forbidden.

### `OccupancyGrid.pack_compact`
- **Does**: Stores only occupied canonical slots plus cell/slot indices and counts.
- **Rationale**: Training does not need to retain hundreds of empty feature vectors per cell.
- **Scaling**: Stable global sorts establish `(cell,u,v,t)` order and vectorized prefix offsets assign
  ranks, avoiding a Python loop over thousands of cells.

### `OccupancyGrid.statistics`
- **Does**: Returns per-cell counts, means, and variances.
- **Rationale**: Mean pooling alone cannot distinguish identical distributions with different mass.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `edit.py` | `GridSpec.cells_for_aabb` returns conservative raster masks | Cell ordering |
| `autoencoder.py` | Cell IDs, compact targets, and occupancy statistics | Indexing or layout |
| Future checkpoints | Grid shape and slots are serialized model semantics | Defaults or indexing |
| Tests | Overflow raises and 45k uniform jewels pack losslessly | Silent truncation |

## Notes

- Canonical within-cell center sorting is a spike simplification. Near-ties can swap ranks; compare
  against Hungarian or optimal-transport matching during actual training.
- Center dimensions remain physical `[-1,1]` coordinates for bucketing even if other features are
  standardized.
