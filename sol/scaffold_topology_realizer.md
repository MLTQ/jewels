# `scaffold_topology_realizer.py`

## Purpose

Bridges autonomous scaffold-conditioned cell counts into the already-selected UCF-trained
stochastic mark flow. This removes privileged fitted birth cells/ranks from the continuation path
without changing or retraining the mark generator.

## Components

### `RealizerTopology`

- **Does**: Carries decoded dense counts plus the corresponding canonical cell/rank vectors.

### `validate_realizer_topology`

- **Does**: Requires an identical grid shape, rejects counts above the frozen flow's rank capacity,
  and expands every positive count into zero-based nested ranks.
- **Rationale**: The topology head uses 1,024 ranks to accommodate initial UCF births, while the
  frozen continuation realizer was trained with 512. Compatibility must be explicit per window;
  silent clipping would confound density and fidelity.

### `predict_realizer_topology`

- **Does**: Rasterizes immutable carried jewels, predicts the next scaffold-conditioned count
  field, and validates it against the realizer contract.
- **Interacts with**: `scaffold_topology_data.py`, `scaffold_topology.py`, and
  `scaffold_topology_eval.py`.

### `realize_topology_marks`

- **Does**: Samples standardized marks for learned cells/ranks, restores physical feature units,
  and hard-projects centers/support starts into their declared topology.
- **Interacts with**: The frozen `BirthMarkFlowModel` and its train-only birth standardizer.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Frozen mark flow | Grid shapes match and every decoded count is at most 512 | Flow topology |
| Learned topology | Counts are one non-negative int64 value per cell | Decode schema |
| Streaming state | Carried features condition prediction but are never modified here | Carry semantics |
| Renderer | Returned marks remain in frontier-local canonical coordinates | Time coordinates |

## Notes

- This bridge supports continuation windows. Initial generation remains a separate compatibility
  problem because the observed initial UCF maximum is 919 ranks in one cell.
- A low oracle-rank overlap does not imply low generated density: Gaussian decompositions are
  non-unique, so all valid predicted ranks must be synthesized and judged by rendering.
