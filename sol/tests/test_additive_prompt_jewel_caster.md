# `test_additive_prompt_jewel_caster.py`

## Purpose

Protects additive probability normalization and irregular continuous free-sampling shapes.

## Components

### `AdditivePromptJewelCasterTests`
- **Does**: Accumulates a tiny two-factor corpus, composes a held-out prompt, and validates normalized
  cell/token distributions plus continuous output.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1c | Every composed cell and cell/role distribution sums to one | Probability contract |
