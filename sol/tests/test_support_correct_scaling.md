# `test_support_correct_scaling.py`

## Purpose

Protects the structural and scaling summaries used by the support-correct stage-1 experiment.

## Components

### `SupportCorrectScalingTests`
- Confirms a field elongated along normalized time reports the expected anisotropy, alignment, and
  five-sigma frame lifespan.
- Confirms the proof summary uses the support-evaluated PSNR curve and applies its initial gates.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `sol/support_correct_scaling.py` | Stable field-structure definitions and compute-curve summary | Metric semantics |
