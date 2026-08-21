# `fit_encoder_teacher_corpus.py`

## Purpose

Builds a resumable support-correct teacher corpus for structural encoder supervision. Unlike the
five-style audit fitter, it preserves individual source ownership and can fit any explicit train or
validation subset without treating a partial failure as a conclusion.

## Components

### `safe_name`
- **Does**: Converts a source ID into a portable checkpoint filename while retaining readable
  identity.

### `select_examples`
- **Does**: Selects an ordered manifest split, optional explicit source list, and optional
  offset/limit slice; rejects missing or empty selections.
- **Rationale**: The corrected scaling manifests already define balanced nested order, so selection
  must not silently reshuffle it.

### `main`
- **Does**: Fits every selected video with the support-complete five-sigma tiled renderer and the
  proven 72k adaptive teacher contract.
- **Does**: Saves a source-owned checkpoint and atomically refreshed compact report after each fit;
  completed checkpoint/report pairs are skipped on restart.
- **Does**: Supports disjoint ordered slices so teacher coverage can expand without refitting a
  completed prefix.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Structural distillation | Checkpoint contains `state` and `source.source_id` | Checkpoint schema |
| Evidence audit | Report freezes ordered source identities and fit protocol | Selection semantics |
