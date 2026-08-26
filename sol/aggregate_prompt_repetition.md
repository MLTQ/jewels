# `aggregate_prompt_repetition.py`

## Purpose

Combines the preregistered one- and two-source exact-prompt neural reports into the Gate 1e
repetition curve. It keeps null-prior comparisons explicit and separates a monotonic data trend
from the final absolute prompt-binding gate.

## Components

### `summarize`

- **Does**: Validates the factorized neural schema and extracts density/token control margins,
  free-running field similarity, prompt retrieval, retained step, and absolute verdict.

### `aggregate`

- **Does**: Applies the frozen nondecreasing checks to correct-versus-null density and token
  margins, correct field match, and retrieval.

### `plot`

- **Does**: Shows density, tokens, independently sampled field match, and retrieval against the
  number of exact-prompt source videos per prompt.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1e report | Factorized prompt gate-v1 reports at unique repetition counts | Input schema |
| Scientific review | Positive margins always mean correct conditioning beat the control | Sign convention |
