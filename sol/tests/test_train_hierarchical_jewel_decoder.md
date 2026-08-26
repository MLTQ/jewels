# `test_train_hierarchical_jewel_decoder.py`

## Purpose

Protects chunked exhaustive prediction/loss helpers, including a non-divisible terminal chunk.

## Components

### `HierarchicalDecoderTrainingTests`
- **Does**: Verifies full output shape and finite loss across chunk boundaries.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Decoder trainer | Evaluation never drops the last partial chunk | Chunk iteration |
