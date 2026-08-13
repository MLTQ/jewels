# `test_scaffold_appearance_adapter_rollout.py`

## Purpose

Protects base ownership and RGB-only mutation across autonomous initial and continuation strides.

## Components

### `ScaffoldAppearanceAdapterRolloutTests`

- **Does**: Confirms the paired frozen field is bit-identical to the standalone rollout for one
  seed, while a nonzero adapter survives three strides without changing topology, IDs, lifecycle,
  geometry, covariance, opacity, or any other non-RGB field.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Four-class render gate | Frozen arm is the existing generator, not a resample | RNG sequence |
| Interactive editor | Counts and append-only IDs match exactly | Topology/row policy |
| Density comparison | Every non-RGB canonical feature matches exactly | Coordinate copy policy |
