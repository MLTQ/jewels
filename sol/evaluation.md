# `evaluation.py`

## Purpose

Measures the tokenizer on windows from held-out source videos using rendered RGB, not feature loss
alone. Uniform continuous `(u,v,t)` samples make the metric inexpensive enough to run during a 2070S
training spike.

## Components

### `evaluate_roundtrip`
- **Does**: Encodes, deterministically decodes, renders target and reconstruction with the all-jewel
  reference, and reports per-window PSNR, macro-by-source PSNR, and decoded-count ratio.
- **Interacts with**: `StructuredJewelAutoencoder`, `FeatureNormalizer`, and `render_exact`.
- **Rationale**: A tokenizer succeeds only if decoded jewels preserve the fitted visual field.

### `EvaluationReport` / `ExampleMetric`
- **Does**: Provide JSON-ready aggregate and per-window measurements, including source identity.
- **Rationale**: Macro-by-source PSNR prevents a video with more fitted windows from dominating the
  model-selection score.

### `select_balanced_examples`
- **Does**: Selects validation windows round-robin across held-out source videos.
- **Rationale**: Sorted corpus paths otherwise fill a small evaluation budget with adjacent windows
  from a single video, hiding source-to-source failures.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train_autoencoder.py` | JSON-ready window/macro-source PSNR and count ratio | Report schema |
| Research comparisons | Fixed seed, exact renderer, and train-only normalization | Evaluation protocol |

## Notes

- This measures tokenizer loss relative to fitted renders, not raw source video quality.
- The trainer's sampled render loss uses this exact renderer on occupied target-corresponding slots;
  this module remains the no-gradient, count-aware held-out measurement.
- A fresh training run records the same protocol at step zero, giving every later score an explicit
  untrained baseline.
- Full-frame perceptual and temporal metrics should be added only after sampled PSNR clears the gate.
