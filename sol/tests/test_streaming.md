# `test_streaming.py`

## Purpose

Protects the persistent-state contract that precedes learned prompt-conditioned continuation.

## Components

### `StreamingTests`

- **Does**: verifies stable birth/carry partitioning, the observed Little's-law identity, complete
  stride coverage, finite-support render equivalence, and argument validation
- **Interacts with**: `streaming.py` and `streaming_metrics.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Continuation data | Carried IDs plus birth IDs exactly equal the active committed state | Ownership semantics |
| Streaming renderer | Each frame is committed once and inactive jewels may be omitted exactly | Support or window semantics |
| Density methodology | Mean active count equals observed birth rate times observed lifespan | Lifecycle accounting |
