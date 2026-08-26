# `test_render_prompt_jewel_programs.py`

## Purpose

Protects non-cherry-picked target selection and completeness checks for prompt-generated Jewel
contact sheets.

## Components

### `PromptProgramRenderTests`

- **Does**: Verifies requested source order, lowest fitter-seed selection, presence of all three
  controls, and generated centroid/token shapes.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Qualitative renderer | One renderable program for each source/control pair | Validation semantics |
