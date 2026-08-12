# `scaffold_topology_rollout.py`

## Purpose

Tests the first autonomous topology sequence without conflating it with mark synthesis. Predicted
counts decide which fitted cell/ranks exist; matching fitted marks are borrowed as an explicit
oracle, carried by stable ID, and never regenerated in later strides.

## Components

### `oracle_matched_birth_mask`

- **Does**: Keeps a target mark exactly when its canonical rank is below the predicted count for
  that cell.
- **Rationale**: This is the highest-quality field available under predicted false negatives while
  leaving false-positive mark synthesis to the frozen realizer coupling gate.

### `rollout_oracle_matched_topology`

- **Does**: Starts at frontier zero, predicts each complete stride from its scaffold and generated
  carry raster, appends only new stable IDs, audits exact carried features, and measures topology
  plus contribution-aware density through the completed frames.
- **Interacts with**: `scaffold_topology_data.py`, `scaffold_topology_eval.py`, `streaming.py`, and
  `splat_density.py`.

### `OracleTopologyRollout`

- **Does**: Returns the retained oracle field, its stable IDs, and a JSON-safe sequential report.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Persistent generation gate | Frontier zero is predicted and later state is carried exactly | Window/state policy |
| Density audit | Metrics cover only complete emission strides | Frame range |
| Visualizer | Returned features remain in global fitted coordinates | Coordinate semantics |
| Scientific interpretation | False-positive ranks are counted but not materialized | Oracle policy |

## Notes

- Oracle-retained density is optimistic: it isolates topology recall and cannot expose artifacts
  from false-positive generated marks.
- Lifecycle and contribution-density eigensolves run on CPU. Batched double-precision `3×3` CUDA
  eigensolves for 20k–72k fields can request multi-gigabyte solver workspaces despite their small
  tensor payload; only the topology network forward pass uses the selected accelerator.
- A passing result licenses coupling decoded counts/ranks to the frozen stochastic mark realizer;
  it is not itself a complete text-to-video result.
