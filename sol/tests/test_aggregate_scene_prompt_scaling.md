# `test_aggregate_scene_prompt_scaling.py`

## Purpose

Protects the frozen Gate 1h data-scaling decision from selective metric reporting.

## Components

### `ScenePromptScalingTests`

- **Does**: Requires token margin, histogram margin, and retrieval to all be nondecreasing.
- **Does**: Keeps positive scaling distinct from the endpoint absolute gate.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1h evidence | All preregistered signals own the scaling verdict | Aggregation semantics |
