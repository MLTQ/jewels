# `scaffold_appearance_adapter.py`

## Purpose

Adds a compact, scaffold-gated RGB correction to a frozen jewel-mark flow.  It tests whether the
washed-out autonomous videos are primarily an appearance-capacity problem without spending another
full 2.13M-parameter flow or allowing color training to disturb motion, density, or identity.

## Components

### `ScaffoldAppearanceAdapter`

- **Does**: Predicts only a three-value RGB velocity residual from the current standardized mark,
  frozen base velocity, addressed causal context/scaffold cells, rank, flow time, and prompt.
- **Rationale**: Direct addressed-cell projections are much smaller than duplicating the base
  flow's 3D encoders.  A zero-initialized head makes an untrained adapter exactly inert.

### `apply_scaffold_rgb_residual`

- **Does**: Applies an external per-cell gate to RGB velocity dimensions while copying every other
  base dimension.
- **Rationale**: Geometry, covariance, opacity, and lifecycle safety are structural properties of
  the computation rather than loss-dependent hopes.

### `sample_appearance_adapted_birth_marks`

- **Does**: Integrates the ordinary frozen base and an RGB-adapted stream from one shared noise
  sample.  The base is evaluated only on its own state; every non-RGB adapted value is copied from
  the new base state after every Euler step.
- **Rationale**: The returned base is bit-identical to ordinary sampling for the same seed, while
  accumulated RGB corrections can condition later adapter steps.

### `top_fraction_cell_gate` / `appearance_feature_loss`

- **Does**: Builds an exact-cardinality saliency gate and scores corrected RGB velocity only on
  selected jewels.
- **Rationale**: This matches the successful top-20% RGB screen and avoids diluting the adapter
  gradient with rows that are guaranteed to remain frozen.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Adapter trainer | Output is a standardized RGB velocity residual | Output units or dimensions |
| Paired rollout | Frozen base sampling is independent of adapted state | Base-state feedback |
| Lifecycle audit | Non-RGB standardized values are bit-identical at every step | Copy policy |
| Saliency control | One external weight per canonical scaffold cell | Gate addressing |
| Checkpoint loader | Architecture is `scaffold_appearance_adapter_v1` | Save schema |

## Notes

- RGB means canonical feature dimensions 9, 10, and 11.  Spatial and temporal color gradients stay
  frozen in this first adapter so a positive result has the narrowest possible interpretation.
- The adapter receives prompt text, but this experiment evaluates reconstruction from a video
  scaffold.  Prompt-only generation remains a later conditioning-transfer gate.
