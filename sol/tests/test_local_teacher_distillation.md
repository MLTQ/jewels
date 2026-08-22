# `test_local_teacher_distillation.py`

## Purpose

Protects the local fitted-teacher extraction, detached correspondence, and independently logged
attribute-loss semantics used by the v3 follow-up experiment.

## Components

### `LocalTeacherDistillationTests`
- **Does**: Verifies canonical teacher attributes and the 2% active-count contract.
- **Does**: Verifies soft correspondence selects local teachers without retaining a center-gradient
  path.
- **Does**: Verifies scale permutation, axis sign, color, gradient, and opacity-mass matches have
  near-zero loss.
- **Does**: Verifies opacity compensation has the declared optical-density interpretation.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Local-distillation trainer | Correspondence is detached and loss keys stay stable | Gradient/key semantics |
| Experiment protocol | Optical mass ratio is teacher active count / student target active count | Ratio definition |
