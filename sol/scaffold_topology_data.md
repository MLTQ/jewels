# `scaffold_topology_data.py`

## Purpose

Builds the sequential discrete-topology contract that a video scaffold must predict. Unlike the
older continuation dataset, it includes the initial stride so a 49-frame LTX clip supplies three
complete 16-frame emission decisions rather than one privileged future.

## Components

### `ScaffoldTopologyView`

- **Does**: Stores the local preceding-stride context, exact carried state, canonical birth
  cells/counts/ranks, both local and global birth marks, and the target active field for one
  complete emission stride.
- **Rationale**: Marks and topology can be swapped independently while stable IDs remain explicit.

### `build_scaffold_topology_views`

- **Does**: Measures global lifecycles and builds all full strides beginning at frontier zero.
- **Interacts with**: `streaming.py`, `streaming_features.py`, and public `pack_births` in
  `streaming_data.py`.
- **Rationale**: The initial state must be generated before later windows can carry it; skipping
  frontier zero would retain the same privileged-prefix loophole as the prior transfer test.

### `rasterize_carried_state`

- **Does**: Samples carried jewel support over the next stride and emits normalized log density,
  occupancy, and mean temporal alpha in the birth-grid order.
- **Rationale**: The topology head can see capacity already supplied by immutable jewels without
  receiving target future births.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Topology trainer | Every full stride, including index zero, owns only new births | Window policy |
| Initial-compatible mark trainer | Frontier zero has empty context; later context is local and causal | Context semantics |
| Frozen mark realizer | Birth cells/ranks use the existing `16×16×8` canonical order | Packing order |
| Sequential rollout | Carried features remain global and bit-identical | Coordinate/ID semantics |
| Topology model | Carry raster channels are log density, occupancy, mean alpha | Channel order |

## Notes

- In current 72k/49-frame and 120k/96-frame fields, most coarse cells are occupied. Count
  allocation is therefore more informative than raw occupied-cell accuracy.
- The selected 1,024-rank budget is required by initial UCF state: its observed per-cell maximum is
  919, versus 283 for later births and 455 for LTX initialization. The older 512-rank mark-realizer
  contract is sufficient for continuation but must be extended or factorized for initial marks.
- Carried support is sampled at temporal cell midpoints. It is conditioning, not a render metric.
- Context covers at most the preceding stride and may include jewels no longer carried into the
  commit. It is empty at frontier zero by construction.
