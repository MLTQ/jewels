# `test_prompt_fit_shards.py`

## Purpose

Protects deterministic balanced sharding and safe staging for local and multi-GPU corpus fitting.

## Components

### `PromptFitShardTests`

- **Does**: verifies four-way class/group balance, disjoint complete ownership, invalid-shard
  rejection, and source-preserving symlink staging
- **Interacts with**: `prompt_fit_shards.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Compute fan-out | Four shards each contain one group from all four classes | Assignment order |
