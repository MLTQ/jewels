# `train_scaffold_appearance_adapter.py`

## Purpose

Trains the compact RGB-only adapter over the selected frozen scaffold mark flow using native-aspect,
high-resolution render coordinates and the successful top-scaffold-saliency gate.

An optional `--teacher-flow` distills the already validated full-flow RGB correction into the
compact adapter.  Real-video render supervision remains active, so distillation transfers the
known-good correction without replacing the visual target.

## Components

### `PreparedAppearanceView`

- **Does**: Keeps one normalized target mark set, causal context, high-resolution-derived scaffold,
  exact fitted carry/background target, binary appearance gate, and saliency weights on the device.

### `_load_guides` / `_prepare`

- **Does**: Decodes video at the declared render resolution before alignment to the jewel grid and
  derives saliency gates from a causal first-stride RGB mean rather than fitted metadata.
- **Rationale**: The 288×192 default preserves the 768×512 corpus's 3:2 aspect ratio.  It removes
  the earlier 40×24 resize's aspect shear while keeping target-only fitted state out of inference
  conditioning.

### `_evaluate_feature_control`

- **Does**: Runs deterministic held-out noise paths and compares frozen/adapted RGB velocity MSE on
  the exact gated rows.  In distillation mode it also reports the full teacher's real-target error
  and the adapter-to-teacher RGB error.
- **Rationale**: This cheap diagnostic catches a failed adapter before the expensive full-video
  renderer, but it is not the final visual acceptance gate.

### `main`

- **Does**: Freezes and hashes the base flow, cycles all training strides, optimizes gated RGB
  velocity plus differentiable high-resolution render patches, and saves a resumable adapter-only
  checkpoint plus periodic immutable step snapshots with full provenance.
- **Distillation path**: A compatible teacher flow is frozen, hash-recorded, and queried on the same
  noisy state.  Only its gated RGB velocity becomes the feature target; base ownership and adapter
  output dimensions remain unchanged.
- **Quiet protection**: The rendered objective includes explicit motion and quiet-region temporal
  error.  Its conservative default weight follows the successful single-field 0.02 screen.
- **Precision**: Training is full FP32 because differentiable Gaussian covariance rendering was
  observed to produce infinite scaled gradients under AMP.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Adapter rollout | Checkpoint architecture is `scaffold_appearance_adapter_v1` | Save schema |
| Frozen base audit | Base SHA-256 and architecture/grid are serialized | Base replacement |
| Teacher distillation | Teacher architecture, standardizers, manifest, and grid match the base | Teacher replacement |
| High-resolution gate | Render width/height preserve 3:2 and patches use that coordinate grid | Aspect policy |
| Saliency comparison | Default gate fraction is exactly 0.20 | Gate policy |
| Leakage-safe evaluation | Optimization uses train sources; diagnostics use validation sources | Split policy |
| Lifecycle ownership | Only adapter parameters receive gradients and only RGB velocity is corrected | Mutation policy |

## Notes

- Fitted carried jewels and backgrounds are supervised render targets only.  They are not serialized
  as generator state and never enter autonomous adapter rollout.
- A passing feature diagnostic is necessary but insufficient; the deterministic four-class
  three-window renderer remains the decision artifact.
