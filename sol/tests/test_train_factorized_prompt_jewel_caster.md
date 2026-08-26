# `test_train_factorized_prompt_jewel_caster.py`

## Purpose

Protects factorized correct/shuffled/null control aggregation for token and continuous-density
metrics.

## Components

### `FactorizedPromptTrainingTests`
- **Does**: Verifies all arms, density NCE, and all three active roles across chunk boundaries.
- **Does**: Verifies that exact-prompt repetition requires both style and action-text identity and
  rejects inconsistent validation-replica metadata.
- **Does**: Verifies that balanced-corpus source allowlists exclude unregistered training fields and
  reject missing registrations.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1b | Controls include density and covariance/surface/gradient NLL | Control schema |
| Gate 1f | Training-source allowlists are strict and source-disjoint | Split selection |
