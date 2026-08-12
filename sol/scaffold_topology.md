# `scaffold_topology.py`

## Purpose

Predicts the discrete birth topology that the frozen jewel-mark realizer previously received from
the fitted target. It treats topology as an occupied-cell decision plus a positive per-cell count;
canonical ranks are then the integers below each emitted count.

## Components

### `ScaffoldTopologyOutput`

- **Does**: Stores occupancy logits and positive-count parameters and exposes differentiable
  positive and expected cell counts.

### `ScaffoldTopologyModel`

- **Does**: Encodes aligned RGB scaffold cells plus immutable carried-state density using local 3D
  convolutions and predicts occupancy and positive count for every `(u,v,t)` cell.
- **Interacts with**: `ContextRasterEncoder` and `ResidualMLP` in `streaming_model.py`.
- **Rationale**: Current coarse grids are almost fully occupied, so the separate heads preserve the
  eventual sparse contract while putting most capacity on density allocation.

### `loss`

- **Does**: Combines per-view balanced occupancy BCE, occupied-cell log-count regression, global
  birth-total calibration, and normalized spatial/temporal mass matching.
- **Rationale**: A low per-cell error can still drift total density or collapse to a position-only
  mean; the global and distribution terms make both failures explicit.

### `decode_counts`

- **Does**: Applies a train-calibrated occupancy threshold, rounds positive counts, and enforces the
  configured per-cell capacity.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Topology trainer | Guide and carry rasters share canonical cell order | Input shape/order |
| Frozen mark realizer | Decoded count `n` materializes ranks `0..n-1` | Count semantics |
| Rollout evaluator | Counts never exceed `GridSpec.slots_per_cell` | Capacity policy |
| Checkpoints | Guide/carry dimensions and grid are serialized | Constructor schema |

## Notes

- Text is deliberately absent. Prompt semantics arrive through the prompt-generated raster; a
  correct/shuffled/null scaffold control measures that causal path directly.
- The default 1,024-rank capacity covers the measured 919-rank UCF initial-cell maximum. Later
  continuation cells fit the frozen realizer's old 512-rank budget; initial mark generation does
  not yet share that narrower contract.
- This model predicts topology only. Oracle-mark and frozen-realizer rollouts are separate gates.
