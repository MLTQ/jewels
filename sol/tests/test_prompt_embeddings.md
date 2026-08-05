# `test_prompt_embeddings.py`

## Purpose

Protects prompt ordering, normalization, provenance, ownership, and atomic cache round trips.

## Components

### `PromptEmbeddingTests`

- **Does**: verifies stable unique text order, canonical manifest hashing, train/validation ownership,
  serialization, and rejection of malformed vectors/order
- **Interacts with**: `prompt_embeddings.py` and `ucf_prompt_manifest.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Prompt training | Every embedding row is unit normalized and bound to exact text | Validation tolerance |
| Held-out templates | Per-example ownership preserves train/evaluation lists and split | Cache schema |
