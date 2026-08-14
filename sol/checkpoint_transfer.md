# `checkpoint_transfer.py`

## Purpose

Loads model-only weights for a compatible scaffold generator while preserving an explicit record
of whether the destination corpus matches or differs from the checkpoint's original manifest.

## Components

### `load_compatible_model_weights`

- **Does**: Validates the architecture name, constructor arguments, grid shape, per-cell rank
  capacity, and source-manifest digest before loading a model state dictionary.
- **Transfer policy**: Same-manifest initialization is the default. A caller must explicitly opt
  into cross-manifest transfer; this changes only model weights and always starts a fresh optimizer
  and gradient scaler.
- **Returns**: Serializable provenance containing source/destination digests, source step, path,
  transfer mode, and the fact that optimizer state was not restored.
- **Interacts with**: The scaffold mark-flow and topology trainers.

### `load_augmented_model_weights`

- **Does**: Loads every shared tensor from a same-manifest base checkpoint while leaving explicitly
  named new module prefixes at their constructor initialization.
- **Rationale**: A zero-residual architectural spike should begin exactly at the selected base
  function without pretending that a partial load is an ordinary compatible checkpoint.
- **Safety**: Base constructor arguments, grid, ranks, manifest, unexpected keys, and missing
  shared tensors are all rejected.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Mark-flow trainer | Model arguments and `16×16×8`, 1,024-rank basis remain exact | Compatibility checks |
| Topology trainer | Model-only transfer never imports optimizer momentum | Restore policy |
| Scientific record | Cross-corpus initialization is named in checkpoint metadata | Provenance schema |
| Coupled-set trainer | Augmentation loads only `set_blocks.*` as new state | Prefix/argument policy |

## Notes

- Resume remains trainer-owned because it intentionally restores optimizer/scaler state and step.
- This module does not reinterpret feature standardizers. Destination trainers compute and save
  statistics from their own training split.
