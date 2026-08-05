# `test_streaming_data.py`

## Purpose

Protects construction of sparse, stable-ID continuation targets from one continuous field.

## Components

### `StreamingDataTests`

- **Does**: verifies carry/birth partitioning, birth-cell capacity, canonical ranks, feature
  normalization round trips, and bounded context rasterization
- **Interacts with**: `streaming_data.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Continuation trainer | Every future-active jewel is exactly carried or newly born | ID ownership |
| Sparse decoder | Count sum equals compact target length and ranks fit capacity | Packing semantics |
