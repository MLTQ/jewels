# `test_plot_prompt_jewel_training.py`

## Purpose

Protects complete, controlled validation-history extraction for the prompt-speaker curve figure.

## Components

### `PromptTrainingPlotTests`

- **Does**: Verifies that all steps and controls are retained and that empty histories are rejected.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Neural prompt curve | Retained checkpoint and every correct/shuffled/null observation | Report parsing |
