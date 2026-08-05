# `streaming_corpus.py`

## Purpose

Builds the multi-clip, prompt-owned continuation corpus from fitted fields. It enforces whole-source
splits and replaces per-video normalization with one shared standardizer fitted only on training
source groups.

## Components

### `PromptedField`

- **Does**: binds a fitted jewel field to class, split, and prompt-cache row ownership

### `PromptedContinuationExample` / `PromptedContinuationCorpus`

- **Does**: expose prompt-aware train/validation examples and shared feature statistics

### `load_prompted_fields`

- **Does**: matches manifest videos to one fitted `w000000` checkpoint across compute shards,
  validates frame/split provenance, and restores canonical jewel features

### `build_prompted_continuation_corpus`

- **Does**: derives every rolling view, fits context/birth standardizers from training views only,
  and installs those identical statistics in both training and validation datasets
- **Rationale**: per-video or validation-fitted normalization leaks held-out information and makes
  prompt comparisons operate in incompatible target spaces

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Prompt trainer | All examples use the same context and birth standardizers | Normalization policy |
| Held-out evaluation | Validation features never affect standardizer moments | Split selection |
| Fit-shard merge | Exactly one checkpoint matches every manifest video stem | Filename contract |
| Prompt sampling | Stored row indices address the validated shared embedding matrix | Ownership schema |
