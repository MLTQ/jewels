# `calibrate_responsibility_distillation.py`

## Purpose

Measures renderer-responsibility target distributions and gradient scales on a declared source and
checkpoint without taking an optimizer step. This makes experiment weights auditable and prevents a
single unstable training launch from defining the objective.

## Components

### `parameter_gradient_norm`
- **Does**: Computes the joint L2 gradient norm of one scalar loss over a declared parameter group.

### `main`
- **Does**: Loads one factorized-v3 checkpoint, source video, and source-owned fitted field.
- **Does**: Uses the same independent active-uniform teacher sample, opacity-sampled student subset,
  support-complete renderer, responsibility moments, and active-count compensation as training.
- **Does**: Records raw loss and geometry/appearance gradient norms plus support/effective counts and
  the fraction of moment targets outside the student's feasible ranges.
- **Does**: Can zero-expand a bounded checkpoint into the residual appearance contract and compare
  bounded versus raw responsibility targets without taking an optimizer step.
- **Does**: Separates the fraction outside the old bounds from the fraction actually projected, so
  a raw renderer-compatible contract reports zero projection without hiding target magnitude.
- **Does**: Reports the fraction of queries that required minimum-Mahalanobis fallback because the
  active-uniform teacher sample contained no jewel inside declared support.
- **Does not**: Construct an optimizer or mutate the checkpoint.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Responsibility protocol | JSON records source/checkpoint ownership and all calibrated components | Report schema |
| Matched GPU screens | Calibration does not take an optimizer step | Adding mutation |
| Factorized-v3 checkpoint | `meta.model_args`, `meta.grid_shape`, and `meta.train_args` are present | Checkpoint schema |
| Expanded appearance protocol | Contract and target-projection mode are recorded in JSON | Report schema |
