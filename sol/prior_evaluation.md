# `prior_evaluation.py`

## Purpose

Provides inexpensive, leakage-safe diagnostics for deciding whether the raster prior is learning
the latent distribution and whether its CLIP condition matters.

## Components

### `evaluate_prior`
- **Does**: Scores identical held-out flow paths with correct, shuffled, and absent conditions;
  measures sampled distribution energy distance; and computes scene-mean and CLIP-retrieval
  baselines. Paired sample MSE remains a secondary diagnostic.
- **Rationale**: A falling training loss alone cannot distinguish generation, memorization, or a
  prior that ignores text.

### `PriorEvaluation`
- **Does**: Supplies a JSON-ready fixed result schema.

### `energy_distance`
- **Does**: Compares two empirical latent distributions using flattened per-feature-scaled Euclidean
  distances, including their within-distribution spread.
- **Rationale**: Unlike paired MSE, energy distance does not reward a collapsed mean over a valid
  independent sample from a one-to-many generator.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Prior trainer | Fixed-seed metrics suitable for checkpoint curves | Evaluation protocol/schema |
| Research decision | Positive conditional gain means correct condition beats no condition | Metric semantics |

## Notes

- Energy distance is still a small-sample latent metric, not a perceptual video metric. The protocol
  must keep sample count and seed fixed across comparisons.
- CLIP retrieval uses only training-source windows.
