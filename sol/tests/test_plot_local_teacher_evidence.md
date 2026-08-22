# `test_plot_local_teacher_evidence.py`

## Purpose

Protects the local-teacher evidence figure from arm-order mistakes and from reversing the meaning
of lower-is-better LPIPS deltas.

## Components

### `LocalTeacherEvidencePlotTests`

- Verifies all sampled metrics come from final held-out evaluation summaries.
- Verifies the exact audit seed-to-arm mapping and both improvement directions.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Evidence figure | Positive LPIPS delta means an improvement | Delta convention |
| Experiment report | Audit seed 0/1/2 maps to control/appearance/full | Candidate order |
