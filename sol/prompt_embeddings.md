# `prompt_embeddings.py`

## Purpose

Defines the durable text-condition sidecar for prompted jewel training. It binds normalized frozen
text vectors to exact prompt strings, examples, splits, encoder identity, and a cryptographic digest
of the source manifest.

## Components

### `collect_prompts` / `manifest_digest`

- **Does**: validates disjoint train/evaluation text, produces stable unique prompt order, and hashes
  the canonical manifest

### `PromptEmbeddingCache`

- **Does**: validates one finite unit vector per prompt and in-range per-example ownership indices
- **Rationale**: anonymous `.npy` vectors cannot prove which text or encoder produced a condition

### `build_prompt_cache`

- **Does**: maps every source example to its training and held-out prompt rows without duplicating
  embeddings

### `save_prompt_cache` / `load_prompt_cache`

- **Does**: atomically persists and validates prompt sidecars

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Prompt encoder | Rows follow `collect_prompts` order and have unit norm | Prompt normalization |
| Multi-clip trainer | Ownership records align with manifest source IDs and splits | Index schema |
| Evaluation | Evaluation prompt indices never occur in each example's training list | Ownership semantics |
| Provenance | Cache digest matches the byte-independent canonical manifest | Digest algorithm |
