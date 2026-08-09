# `tokenizer_checkpoint.py`

## Purpose

Centralizes tokenizer construction from versioned checkpoint metadata so evaluation and rendering
cannot silently restore a checkpoint with the wrong latent contract.

## Components

### `build_tokenizer`
- **Does**: Dispatches the declared architecture ID to its model class and injects the audited grid
  specification.
- **Interacts with**: Sparse variable-count and occupied-group tokenizers.
- **Rationale**: All artifact consumers must share one architecture registry as tokenizer variants
  are compared.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Evaluators/renderers | Unsupported architecture IDs fail explicitly | Dispatch behavior |
| Checkpoints | `model_args` excludes `spec`; grid metadata supplies it | Metadata schema |
