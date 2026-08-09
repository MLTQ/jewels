# `test_streaming_model.py`

## Purpose

Protects the sparse conditional birth-model tensor and variable-count contracts.

## Components

### `StreamingModelTests`

- **Does**: verifies global and cell-local prefix encoding, text/null/dropout paths, exact-rank
  training output, finite loss, variable-count decode, sparse occupied/empty count balancing, and
  invalid input rejection
- **Interacts with**: `streaming_model.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Continuation trainer | One compact output row per requested birth rank | Sparse decode schema |
| Sampling | Materialized birth count equals the sum of predicted cell counts | Count semantics |
| Spatial continuation | Local mode emits exactly one context token per birth cell | Context layout |
| Prompt controls | Dropped text exactly matches the learned null path; correct text remains distinct | Dropout semantics |
