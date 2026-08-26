# `test_render_exact_prompt_sources.py`

## Purpose

Protects exact style/action matching and source-count provenance for exact-prompt source sheets.

## Components

### `ExactPromptSourceRenderTests`

- **Does**: Rejects style-only/action-only matches, requires at least two new source IDs, and accepts
  larger registered repetition sets.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1e provenance image | Matching uses exact style and full action sentence | Source ownership |
