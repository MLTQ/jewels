# `plot_prompt_scene_ablation.py`

## Purpose

Creates the causal evidence panel comparing the balanced independent-Jewel prompt speaker with the
otherwise matched shared-scene speaker.

## Components

### `extract_ablation`

- **Does**: Extracts held-out token/density controls, free-running histogram controls, retrieval,
  and the correct-minus-null free-generation margin from both registered reports.

### `plot_ablation`

- **Does**: Shows that shared scene state reverses the free-generation margin while retaining the
  frozen 0.02 gate line and failed absolute verdict.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Pitch evidence | Balanced Gate 1f and shared-scene Gate 1g reports | Report schemas |
| Scientific review | Failure remains visible; bars are not renormalized | Plot semantics |
