# `test_tokenizer_checkpoint.py`

## Purpose

Protects explicit tokenizer checkpoint dispatch as representation variants are introduced.

## Components

### `TokenizerCheckpointTests`
- **Does**: Verifies restoration of both supported tokenizer families and rejection of unknown
  architecture identifiers.
- **Interacts with**: `tokenizer_checkpoint.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Artifact consumers | Architecture IDs map to exactly one model class | Registry mappings |
