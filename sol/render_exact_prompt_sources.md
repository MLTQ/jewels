# `render_exact_prompt_sources.py`

## Purpose

Shows that Gate 1e's repeated prompts own genuinely distinct source videos rather than fitter
replicas or duplicated pixels. Each row compares the held-out target with every newly generated
video that shares its exact style/action text.

## Components

### `match_sources`

- **Does**: Matches by exact `(style, source_prompt)` and requires exactly one held-out target plus
  at least two new source IDs per prompt.

### `main`

- **Does**: Loads the middle frame from every source at a common resolution and writes a labeled,
  repetition-scaled contact sheet.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Exact-prompt provenance | Fit manifest contains at least two new exact-text videos per prompt | Manifest schema |
| Scientific review | Target and training sources remain visibly and textually distinct | Matching semantics |
