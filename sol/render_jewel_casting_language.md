# `render_jewel_casting_language.py`

## Purpose

Produces the qualitative counterpart to Gate 0. For one independently optimized field per held-out
video, it renders early, middle, and late frames from the continuous source, motif-only cast,
half-residual cast, exact full-residual cast, and deliberately lattice-locked negative control.

## Components

### `select_qualitative_records`
- **Does**: Selects the lowest fit seed for every registered validation source from the largest
  vocabulary's rows.
- **Rationale**: Source selection comes from the frozen report rather than a visually preferred
  post-hoc example.

### `_load_codebook`
- **Does**: Restores the learned motif prototypes and train-owned normalization contract.

### `main`
- **Does**: Decodes all registered residual arms, renders identical frame coordinates with the
  support-complete renderer, and writes a labeled contact sheet.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Qualitative evidence | Gate-v0 record paths and colocated largest-vocabulary codebook | Report layout |
| Scientific review | Same source, time samples, renderer, and background for every arm | Candidate/render ownership |
