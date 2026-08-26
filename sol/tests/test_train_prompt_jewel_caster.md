# `test_train_prompt_jewel_caster.py`

## Purpose

Protects the exact compositional prompt string and correct/shuffled/null teacher-forced control
aggregation used by Gate 1.

## Components

### `PromptCasterTrainingTests`
- **Does**: Verifies style/action text ownership and complete arm/role metrics across chunked data.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1 | Prompt text contains style and source sentence; controls include every active role | Control schema |
