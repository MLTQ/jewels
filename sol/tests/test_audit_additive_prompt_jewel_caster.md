# `test_audit_additive_prompt_jewel_caster.py`

## Purpose

Protects nearest-factor text resolution used to keep Gate-1c inference free of categorical IDs.

## Components

### `AdditivePromptAuditTests`
- **Does**: Verifies normalized cosine nearest-neighbor resolution.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1c | Resolved indices derive only from frozen text-vector similarity | Resolver contract |
