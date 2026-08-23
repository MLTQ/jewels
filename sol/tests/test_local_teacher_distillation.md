# `test_local_teacher_distillation.py`

## Purpose

Protects the local fitted-teacher extraction, detached correspondence, and independently logged
attribute-loss and renderer-responsibility semantics used by the v3 follow-up experiments.

## Components

### `LocalTeacherDistillationTests`
- **Does**: Verifies canonical teacher attributes and the 2% active-count contract.
- **Does**: Verifies soft correspondence selects local teachers without retaining a center-gradient
  path.
- **Does**: Verifies scale permutation, axis sign, color, gradient, and opacity-mass matches have
  near-zero loss.
- **Does**: Verifies opacity compensation has the declared optical-density interpretation.
- **Does**: Verifies active-uniform sampling excludes inactive jewels so opacity is applied once.
- **Does**: Verifies Mahalanobis responsibility retains elongated support that center distance would
  reject, and keeps its targets detached from student centers.
- **Does**: Verifies analytic single-teacher color/Jacobian targets and zero-loss moment matching.
- **Does**: Verifies out-of-support queries are explicitly marked when nearest-Mahalanobis fallback
  is required.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Local-distillation trainer | Correspondence is detached and loss keys stay stable | Gradient/key semantics |
| Experiment protocol | Optical mass ratio is teacher active count / student target active count | Ratio definition |
| Responsibility gate | Target covariance, color, gradients, and diagnostics retain composited meanings | Moment schema |
