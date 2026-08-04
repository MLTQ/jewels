# `hierarchical_inpaint.py`

## Purpose

Bridges fine 32³ cursor edits to the validated 16³ PCA hierarchy. A coarse code is regenerated
when any of its eight fine cells is touched; every other code remains bit-identical.

## Components

### `coarsen_dirty_mask`
- **Does**: Conservatively reduces fine dirty cells with an `any` operation over each 2³ block.
- **Rationale**: One PCA code jointly controls all cells in its block, so partial-code edits are not
  representable.

### `expand_coarse_mask`
- **Does**: Expands the block mask to all fine cells that may change after PCA decoding.
- **Rationale**: Jewel merging must replace the entire decoded block, not only the original cursor
  footprint.

### `hierarchical_masked_flow_inpaint`
- **Does**: Runs clamped flow sampling on dirty coarse codes and returns both mask resolutions.
- **Interacts with**: `EditPlan`, `BlockPCACodec`, and `masked_flow_inpaint`.

### `restore_clean_codes`
- **Does**: Reapplies original raw coarse codes after latent de-normalization.
- **Rationale**: Normalize/de-normalize arithmetic can introduce ~1e-5 float error even when the
  normalized code was clamped exactly; the editor contract is stricter than that.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Editor bridge | Fine and coarse masks use canonical `(u,v,t)` raster order | Mask layout |
| PCA decoder | All eight cells of a dirty code are treated as replaceable | Expansion policy |
| Tests | Clean normalized and raw coarse codes are exactly preserved | Clamp semantics |

## Limitations

The current prior was trained for full generation, not masked repair or protected-jewel
conditioning. This module proves the locality and reconstruction contract; visual boundary quality
must be measured and then improved with masked fine-tuning.
