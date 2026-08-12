# `scaffold_mark_rollout.py`

## Purpose

Runs the first honest prompt-scaffold-to-jewel sequence: an empty-state initial stride followed by
continuations whose topology and marks see only the append-only model-generated jewel field.

## Components

### `ScaffoldMarkRollout` / `ScaffoldMarkWindowReport`

- **Does**: Returns the generated global field, contiguous stable IDs, per-stride decoded counts,
  capacity, state sizes, and exact prior/carry feature audits.

### `rollout_scaffold_marks`

- **Does**: Selects causal context/carry, predicts topology, samples all declared ranks, converts
  marks to global time, and appends new stable IDs for each scaffold stride.
- **Interacts with**: `scaffold_mark_data.py`, `scaffold_topology_realizer.py`, and
  `streaming_features.py`.
- **Rationale**: Recomputing context from generated lifecycles closes the teacher-forced fitted
  prefix loophole while immutable rows preserve interactive jewel identity.
- **Clip-boundary contract**: The first stride permits time-cell-zero jewels whose finite support
  began before frame zero. Later strides project every birth strictly into its declared local time
  cell, so the exception cannot leak across generated continuation boundaries.
- **Matched control**: `allow_initial_prefrontier=False` restores the former strict projection for
  a deterministic boundary ablation without changing continuation behavior.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Three-window render gate | First guide starts from exactly zero jewels | Initial-state policy |
| Persistent editor | Existing feature rows and stable IDs never change | Append semantics |
| Topology head | Carry raster is derived from generated active marks | State ownership |
| Mark flow | Every decoded rank is synthesized; no clipping or oracle marks | Capacity policy |
| Boundary ablation | Only the first local time cell changes under the strict control | Projection policy |

## Notes

- The rollout retains emitted jewels after they become inactive; finite support makes their later
  render contribution exactly zero under the truncated reference contract.
