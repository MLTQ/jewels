# `test_perceptual_eval.py`

## Purpose

Covers the metric-independent scoring path with an injected callable, so the suite never needs
the `lpips` package or its pretrained weights.

## Components

### `test_score_arms_reports_per_frame_and_mean`
- **Does**: An identical arm must score exactly zero and a perturbed arm strictly positive, with
  per-frame scores and the standard render signature present.

### `test_score_arms_rejects_shape_mismatch`
- **Does**: Arms that do not share the target's shape are refused rather than silently resized.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `perceptual_eval.py` | `score_arms` stays pure and metric-injectable | Signature change |
