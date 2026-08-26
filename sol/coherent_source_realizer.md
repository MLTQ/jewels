# `coherent_source_realizer.py`

## Purpose

Provides the Gate 2a8 causal upper bound in which one source-level token owns every addressed block
in a window. It tests whether the block oracle failed because independent local choices destroyed
cross-block and temporal coherence.

## Components

### `CoherentSourceRealizer`

- **Does**: Selects one eligible training field by mean addressed descriptor distance and emits its
  complete continuous-centroid, active-token Jewel program.
- **Does**: Restricts semantic scene arms to their six registered training fields while the null arm
  can choose from all 18.
- **Rationale**: Holding one source choice across all blocks is the smallest control that restores
  every source-owned dependency without changing the Gate 0f physical vocabulary.
- **Does**: Pools the four nearest eligible source programs for addressed active-token likelihood,
  while generation remains a single-source choice.

### `fit_coherent_source_realizer`

- **Does**: Precomputes normalized addressed descriptors and quantized physical tokens for each
  source-owned field.
- **Rationale**: The source-disjoint target may select a training program but can never contribute
  Jewel rows to it.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a8 audit | Exactly one selected field per generated window | Per-block selection |
| Gate 0f decoder | Continuous centroids plus three active K=1,024 tokens | Output schema |
| Scientific review | Retrieval is labeled as a causal ceiling, never prompt inference | Claim ownership |
