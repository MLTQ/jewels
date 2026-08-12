# `scaffold_mark_data.py`

## Purpose

Builds the leakage-safe 1,024-rank mark corpus spanning initial and continuation strides, and
reconstructs the same causal context/carry selections from a model-generated field at inference.

## Components

### `ScaffoldMarkCorpus` / `ScaffoldMarkSource`

- **Does**: Couples prompted fields to every complete topology view and shares context/birth
  standardizers fitted only on training sources.
- **Interacts with**: `scaffold_topology_data.py` and `streaming_corpus.py`.

### `build_scaffold_mark_corpus`

- **Does**: Builds frontier-zero plus all later full strides and enforces source/class split safety.
- **Rationale**: A continuation-only normalizer or rank basis would preserve the fitted-seed
  loophole and leave high-rank initial cells out of distribution.

### `rasterize_scaffold_context`

- **Does**: Produces the existing 46-channel context raster, with an explicit all-zero initial
  raster when no jewel has yet been emitted.

### `generated_window_state`

- **Does**: Selects preceding-stride context, active carry, and active commit rows from an
  append-only generated field using measured finite-support lifecycles.
- **Rationale**: Later strides must condition on model-produced marks without borrowing fitted IDs.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Scaffold mark trainer | Birth normalization includes initial and continuation marks | Split/window policy |
| Autonomous rollout | Frontier zero returns empty context/carry | Empty-state semantics |
| Mark flow | Context raster channels remain `[mean, variance, log_count, occupied]` | Raster order |
| Stable-ID audit | Returned indices address rows in the append-only generated field | Reordering |

## Notes

- The context window is one stride, matching the topology dataset; carried jewels are selected
  independently by whether they remain active in the next commit.
