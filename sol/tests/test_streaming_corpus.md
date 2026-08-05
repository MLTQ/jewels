# `test_streaming_corpus.py`

## Purpose

Protects shared train-only normalization and prompt-class split coverage for multi-clip continuation.

## Components

### `StreamingCorpusTests`

- **Does**: proves extreme validation features cannot change shared moments, both splits receive the
  same standardizer objects, and every class must occur in train and validation
- **Interacts with**: `streaming_corpus.py` and `streaming_data.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Held-out prompt gate | Validation statistics never influence training normalization | Fit population |
| Multi-clip model | All videos share one normalized target coordinate system | Per-example replacement |
