# `test_render_scaffold_mark_rollout.py`

## Purpose

Protects the target-free causal background baseline and explicit multi-window seam measurements.

## Components

### `RenderScaffoldMarkRolloutTests`

- **Does**: Verifies background color comes only from supplied initial frames and seam changes are
  measured exactly at stride boundaries. Full rollout reports separately retain per-frontier and
  first-stride density vectors.
- **Interacts with**: `render_scaffold_mark_rollout.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Autonomous render gate | Background never reads a fitted checkpoint | Background input |
| Seam audit | Boundaries are temporal differences at `stride-1` | Index convention |
