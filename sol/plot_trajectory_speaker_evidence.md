# `plot_trajectory_speaker_evidence.py`

## Purpose

Consolidates the native Jewel grammar experiments into a pitch-facing quantitative figure and a
paired qualitative proof sheet.

## Components

### `load_evidence`

- **Does**: Extracts prompt-control margins, retrieval, scene consistency, and learned-speaker
  convergence from the frozen JSON reports.

### `plot_evidence`

- **Does**: Plots the causal recognizability ladder, rendered prompt margins, semantic retrieval,
  and correct/shuffled/null training curves.

### `build_proof_sheet`

- **Does**: Crops the same middle frame from exact-compiler and learned-speaker seed sheets, pairing
  correct and cyclic-shuffled generations for all three prompt classes.
- **Rationale**: The sheet makes the causal class swap inspectable without cherry-picking different
  frames or seeds across arms.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Evidence report | Frozen result schemas and qualitative sheet layout | Source artifact schema |
| Visual comparison | Middle frame and matched seed within each experiment | Crop ownership |
| Claim scope | Manual recognizability counts are labeled visual review, not a numeric model metric | Legend semantics |
