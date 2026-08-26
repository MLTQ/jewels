# `plot_prompt_jewel_training.py`

## Purpose

Makes the neural prompt-speaker plateau visible. The figure shows every registered validation
checkpoint for correct, shuffled, and prompt-blind null inputs, so later degradation cannot be
mistaken for insufficient training.

## Components

### `load_curve`

- **Does**: Validates each neural report schema and extracts aligned token plus centroid/density
  validation histories and the actually retained checkpoint.

### `plot`

- **Does**: Plots all three controls for the joint-text and factorized-text speakers. Lower values
  are better and the retained checkpoint is labeled directly.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Prompt feasibility report | Gate 1a and Gate 1b history/control schemas | Input report schemas |
| Scientific review | Curves include all recorded steps and all controls | Selection semantics |
