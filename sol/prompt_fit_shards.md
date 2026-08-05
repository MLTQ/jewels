# `prompt_fit_shards.py`

## Purpose

Partitions the fixed prompt corpus into deterministic, class-balanced fitting shards. It supports a
small local preflight and later multi-GPU fan-out without changing example ownership or prompts.

## Components

### `select_fit_shard`

- **Does**: sorts by class and source group, then assigns source-group positions round-robin
- **Rationale**: with four shards and four groups, every shard contains one group from every class

### `stage_fit_shard`

- **Does**: creates idempotent non-overwriting symlinks for only the selected videos

### `main`

- **Does**: writes a shard manifest carrying the full source-manifest digest and staged example
  records

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Preflight fitting | Four shards yield four classes and one group per shard | Sort/stride policy |
| Multi-GPU fitting | Shards are disjoint and their union equals the source manifest | Assignment algorithm |
| Result merge | Every shard records the same source-manifest digest | Report schema |
