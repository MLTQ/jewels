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
  log-count per cell, and decodes only requested canonical ranks
- **Rationale**: variable birth count is part of generation; padded existence fields are unnecessary
- **Rationale**: local context mode preserves spatial correspondence between preceding jewels and
  the birth cells that extend them; global mode remains load-compatible with the first checkpoint

### `forward_training` / `loss`

- **Does**: evaluates exact target ranks and balances standardized feature reconstruction with
  log-count reconstruction

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
| Prompted successor | Context embedding can later receive text cross-attention/conditioning | Removing conditioning path |

## Notes

- `context_mode="global"` preserves the first checkpoint's semantics. New training defaults to
  `context_mode="local"` because the global bottleneck lost spatial detail in visual continuation.
- Passing correct versus disjoint shuffled/null prefixes remains a required selectivity control.
