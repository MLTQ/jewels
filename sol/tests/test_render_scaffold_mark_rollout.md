# `test_render_scaffold_mark_rollout.py`

## Purpose

Protects the target-free causal background baseline and explicit multi-window seam measurements.

## Components

### `RenderScaffoldMarkRolloutTests`

- **Does**: Verifies background color comes only from supplied initial frames and seam changes are
  measured exactly at stride boundaries. It also protects the optional frozen-base panel without
  changing the established baseline panel list and verifies the static-detail screen excludes
  opacity and temporal RGB gradient. It also checks the scaffold saliency gate selects exactly the
  declared cell fraction and that class seeds derive from the complete validation order rather than
  a filtered high-resolution subset. Full rollout reports separately retain per-frontier and
  first-stride density vectors. A base-lock unit also rejects any augmented checkpoint that
  changes a shared tensor outside its declared new-module prefix.
- **Interacts with**: `render_scaffold_mark_rollout.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Autonomous render gate | Background never reads a fitted checkpoint | Background input |
| Seam audit | Boundaries are temporal differences at `stride-1` | Index convention |
| Filtered evaluation | Every source keeps its full-split deterministic seed | RNG convention |
| Coupled-set attribution | Every shared base tensor remains bit-identical | Checkpoint ownership |
