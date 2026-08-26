# `test_prompt_jewel_caster.py`

## Purpose

Protects active-token encoding/decoding, exact continuous-center ownership, histogram dimensions,
joint loss, factorized style/action conditioning, and chunked free-sampling shapes for native prompt
casters.

## Components

### `PromptJewelCasterTests`
- **Does**: Fits a tiny vocabulary and exercises both teacher-forced and prompt-only paths without a
  pretrained text encoder or GPU.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1 | Three active token columns and continuous sampled centers | Output language schema |
| Inference audit | Free sampling takes text plus declared generator only | Sampling signature |
