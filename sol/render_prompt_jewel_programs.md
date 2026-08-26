# `render_prompt_jewel_programs.py`

## Purpose

Provides the qualitative counterpart to the prompt-caster controls. For each report-owned held-out
prompt, it renders early, middle, and late frames from the independently fitted target field and
from correct-prompt, shuffled-prompt, and prompt-blind generated Jewel programs.

## Components

### `select_target_records`

- **Does**: Selects the lowest fitter seed for every registered validation source without visual
  cherry-picking.

### `validate_programs`

- **Does**: Requires all three prompt controls and verifies their continuous-centroid and
  three-token shapes before rendering.

### `main`

- **Does**: Decodes prompt-only programs through the frozen bundle-1 codebook and renders identical
  time coordinates, target-owned background, and support-complete renderer for every arm.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Qualitative prompt evidence | Additive report, generated-program map, frozen Gate 0f codebook | Input schemas |
| Scientific review | Report-owned source order and identical frame probes across controls | Selection/render semantics |
