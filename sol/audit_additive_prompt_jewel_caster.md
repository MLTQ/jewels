# `audit_additive_prompt_jewel_caster.py`

## Purpose

Fits and audits Gate 1c's source-disjoint additive style/action language. It resolves factor strings
through frozen BGE, composes shrunken count distributions, free-runs 72k Jewels, and retains the same
correct/shuffled/null prompt controls.

## Components

### `resolve_factor`
- **Does**: Maps a supplied text vector to the nearest declared style or action factor phrase; the
  audit verifies exact resolution for every held-out prompt.

### Count fitting and teacher-forced controls
- **Does**: Samples balanced source fields, accumulates global/style/action counts, and compares cell
  plus per-role token NLL on all nine validation fields.

### Prompt-only generation and gate
- **Does**: Samples irregular programs from resolved text factors, renders diagnostics, compares
  target histograms/top-1 retrieval, and audits inference inputs and finite output.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 1c protocol | Frozen BGE resolver and Dirichlet/product composition | Statistical ownership |
| Scientific review | Training counts exclude every validation source | Data ownership |
| Pitch artifacts | Additive caster/program checkpoints and v1 report | Artifact schema |
