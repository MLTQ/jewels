# `streaming_model.py`

## Purpose

Defines the first learned continuation model: a prefix-conditioned sparse birth process. It predicts
new jewel counts and marks while carried jewel IDs and parameters remain outside the model.

## Components

### `ContextRasterEncoder`

- **Does**: encodes bounded prefix count/moment rasters with residual 3D convolutions and returns
  either a legacy global vector or aligned per-cell tokens with global context
- **Interacts with**: `rasterize_context` in `streaming_data.py`

### `BirthContinuationModel`

- **Does**: combines prefix context with factorized `(u,v,birth-time)` embeddings, predicts one
  log-count per cell, optionally injects a frozen text embedding, and decodes only requested
  canonical ranks
- **Rationale**: variable birth count is part of generation; padded existence fields are unnecessary
- **Rationale**: local context mode preserves spatial correspondence between preceding jewels and
  the birth cells that extend them; global mode remains load-compatible with the first checkpoint

### `forward_training` / `loss`

- **Does**: evaluates exact target ranks and balances standardized feature reconstruction with
  log-count reconstruction; optional per-example text dropout trains a learned null condition
- **Sparse count control**: `balance_count=True` averages occupied and empty cell errors separately,
  preventing thousands of empty birth cells from overwhelming prompt-dependent topology.

### `forward_from_context`

- **Does**: evaluates the same target ranks from a cached, shuffled, or nulled context embedding
- **Rationale**: correct/shuffled/null controls must share identical targets and model weights

### `decode`

- **Does**: rounds predicted counts and materializes only emitted birth marks

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Continuation trainer | Context raster has 46 channels and target features are standardized | Input schema |
| Evaluator | Birth features are frontier-local and carried features are merged separately | Output semantics |
| Prompted successor | `text_dim > 0` enables text projection and a trainable null condition | Text/dropout semantics |

## Notes

- `context_mode="global"` preserves the first checkpoint's semantics. New training defaults to
  `context_mode="local"` because the global bottleneck lost spatial detail in visual continuation.
- Passing correct versus disjoint shuffled/null prefixes remains a required selectivity control.
- Text dropout must be used during prompt training. Calling the learned null branch without dropout
  training does not produce a meaningful unconditional control.
