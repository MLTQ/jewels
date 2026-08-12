# `test_audit_prompted_washout.py`

## Purpose

Protects the failure-mode decomposition used to decide whether topology calibration can plausibly
fix prompted video washout.

## Components

### `PromptedWashoutAuditTests`
- **Does**: Verifies exact feature-group hybrids, detection of spatial cell escape, and
  target-relative loss of visible detail, including low SSIM for a conditional-mean image.
- **Interacts with**: `audit_prompted_washout.py` and `GridSpec`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Research audit | Hybrids do not silently modify unrelated parameters | Feature slices |
| Topology decision | Assigned-cell escape is measured independently of count error | Cell convention |
| Visual interpretation | A constant conditional mean has near-zero detail ratios | Metric semantics |
