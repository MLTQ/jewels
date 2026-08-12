# `scaffold_topology_eval.py`

## Purpose

Evaluates whether a scaffold predicts fitted birth density rather than merely replaying the nearly
full coarse-grid prior. It uses correct, cross-class shuffled, null, no-carry, and train-mean
controls with one threshold calibrated only on training fields.

## Components

### `TopologyControlView`

- **Does**: Binds one source/stride to its guide, immutable carry raster, and target count field.

### `topology_metrics`

- **Does**: Reports cell/count error, occupancy precision/recall/F1/IoU, rank-slot
  precision/recall/F1, global count ratio, and per-view count correlation.
- **Rationale**: At current resolution occupancy alone is close to trivial; slot overlap and count
  correlation expose density-field quality.

### `expand_topology_counts`

- **Does**: Converts one dense integer count per cell into the nested canonical `(cell, rank)`
  vectors consumed by the frozen mark realizer.
- **Rationale**: Learned topology must enter the realizer through the same rank convention used by
  oracle targets, with explicit capacity validation rather than silent truncation.

### `calibrate_occupancy_threshold`

- **Does**: Chooses among fixed thresholds using training slot F1 and count MAE only.
- **Rationale**: Validation topology must not tune the discrete emission rule.

### `evaluate_topology_controls`

- **Does**: Holds target and carry fixed while replacing the raster with a different-class or zero
  scaffold, also measuring correct guide without carry and a per-stride train-mean baseline.
- **Rationale**: Correct versus shuffled/null is the causal prompt-scaffold test; train mean catches
  position-only solutions.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Topology trainer | Threshold is selected from training outputs only | Calibration policy |
| Research gate | Shuffled guides come from another class at the same stride index | Control policy |
| Rollout | Slot recall equals the fraction of oracle target ranks available to retain | Metric definition |
| Mark realizer | Expanded ranks are contiguous from zero inside every occupied cell | Rank convention |
| Result JSON | Aggregate and per-class controls use stable metric names | Schema |

## Notes

- `predicted_births` and `target_births` are serialized as floats for uniform JSON conversion even
  though decoded counts are integral.
