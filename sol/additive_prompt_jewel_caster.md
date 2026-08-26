# `additive_prompt_jewel_caster.py`

## Purpose

Implements a deliberately low-capacity prompt-to-Jewel language model. It composes shrunken global,
style-text, and action-text count posteriors, then samples continuous irregular positions and active
Jewel marks. It cannot memorize held-out style/action interactions.

## Components

### `AdditiveLanguageCounts` / `accumulate_language_counts`
- **Does**: Accumulates source-disjoint cell and cell/role/token counts for global, style, and action
  factors.

### `AdditivePromptJewelCaster.probabilities`
- **Does**: Shrinks style/action posteriors toward the global Dirichlet prior and composes a new
  combination by normalized product of experts.

### `negative_log_likelihood` / `sample`
- **Does**: Evaluates factor controls and free-runs cell counts, uniform within-cell continuous
  jitter, and all three active marks.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1c | Token concentration 64, cell concentration 256, product composition | Statistical protocol |
| Prompt inference | Resolved style/action factor indices originate from text embeddings only | Input ownership |
| Renderer | Sampled centers are continuous within addressed cells, never fixed centers | Irregularity contract |
