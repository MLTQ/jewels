# `birth_set_coupling.py`

## Purpose

Adds linear-memory coordination among jewels born in the same cell and adjacent space-time cells.
It addresses the independent-rank speckle failure without changing topology, rank count, persistent
carry, IDs, feature units, or the stochastic flow path.

## Components

### `rasterize_set_moments`

- **Does**: Pools learned hidden rows into per-cell mean, variance, log-count, and occupancy.
- **Rationale**: Learned set statistics expose relations that the raw 22-D noisy-mark raster cannot
  express, while scatter pooling remains linear in the number of jewels.

### `NeighborhoodBirthSetBlock`

- **Does**: Encodes hidden set moments with a 3D neighborhood convolution and returns a shared
  residual to every addressed jewel.
- **Rationale**: Jewels in one cell need a shared local composition state; adjacent cells also need
  continuity across artificial grid boundaries. Full pairwise attention would scale poorly for
  initial cells containing hundreds of ranks.
- **Initialization**: The final residual projection is exactly zero, so adding the block to a
  frozen checkpoint leaves every predicted velocity bit-identical before training.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `birth_mark_flow.py` | Output shape and row order equal the input hidden set | Shape/order |
| Augmented checkpoint loader | New state keys live below `set_blocks.` | Module name |
| Coupled trainer | Pooling is permutation-equivariant and differentiable | Set semantics |
| Autonomous rollout | No rows, cells, ranks, counts, or IDs are created here | Ownership |

## Notes

- Empty cells retain zero input statistics but can receive context from occupied neighbors.
- One block with raster depth zero is the first bounded gate; deeper stacks require evidence.
