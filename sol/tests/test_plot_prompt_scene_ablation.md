# `test_plot_prompt_scene_ablation.py`

## Purpose

Protects the causal shared-scene plot from hiding a failed absolute gate or reversing margin signs.

## Components

### `PromptSceneAblationTests`

- **Does**: Confirms a negative-to-positive correct-minus-null margin is recorded as improvement but
  not relabeled as an absolute pass.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Shared-scene evidence | Margin is `correct - null`; report gate owns pass/fail | Extraction semantics |
