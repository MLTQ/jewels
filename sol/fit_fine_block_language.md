# `fit_fine_block_language.py`

## Purpose

Fits Gate 2a6's fine 16x16x8/K=1024 local vocabulary and writes its ordered 2,048-token training
programs without training another conditionally independent neural decoder.

## Components

### `validate_fine_language_settings`

- **Does**: Enforces the preregistered routing shape and vocabulary size.

### Fine-language fit

- **Does**: Reuses the immutable 18/9 split and Gate 0f physical normalizer, fits 20 deterministic
  Lloyd iterations over all 36,864 training blocks, and encodes every training program.
- **Does**: Stores portable prototypes, time-major Morton order, source alignment, and decision-count
  evidence for the downstream scene/block constellation oracle.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a6 protocol | 16x16x8, K=1024, 20 iterations | Fit settings |
| Scene/block oracle | Checkpoint keys match prior block-language checkpoints | Artifact schema |
| Future transformer | Exactly 2,048 time-major Morton block tokens per window | Program shape/order |
