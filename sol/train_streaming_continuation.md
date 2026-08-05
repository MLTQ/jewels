# `train_streaming_continuation.py`

## Purpose

Trains the first learned persistent continuation overfit from one 96-frame joint fit. Four
32-prefix/16-future views share one sparse birth model; carried jewels never enter the optimizer.

## Components

### `PreparedView`

- **Does**: keeps bounded context rasters and normalized sparse birth targets resident on device

### `main`

- **Does**: loads the monolithic checkpoint, constructs stable-ID views, trains cyclically with
  cosine decay, uses aligned per-cell prefix context by default, evaluates
  correct/shuffled/null prefix controls, and saves resumable checkpoints
- **Interacts with**: `streaming_data.py`, `streaming_model.py`, and
  `streaming_continuation_eval.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment record | Checkpoint stores source path, grid, time contract, and both standardizers | Metadata schema |
| Prompt phase | Correct prefix must beat shuffled/null before text conditioning begins | Evaluation fields |
| Recovery | `--resume` restores model, optimizer, scaler, and next step | Checkpoint keys |

## Notes

- This is intentionally an overfit gate on four targets, not a generalization claim.
- Default 256 slots exceeds the observed maximum birth-cell occupancy of 147.
- `--context-mode global` reproduces the first global-pooling architecture; `local` is the default
  after visual evaluation showed that a single vector could not preserve spatial continuation.
