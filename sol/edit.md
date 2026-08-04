# `edit.py`

## Purpose

Translates a cursor operation into the constraints required by a local generative repair: untouched
context, protected moved jewels, and dirty raster cells covering the vacated region, destination,
and straight-line sweep between them.

## Components

### `EditPlan`
- **Does**: Carries source/destination geometry, masks, clean context, and protected moved jewels.
- **Interacts with**: `masked_flow_inpaint` in `inpaint.py` and structured decode in `token_grid.py`.

### `EditPlan.merge`
- **Does**: Reassembles untouched context, generated dirty-region jewels, and protected moved jewels.
- **Rationale**: Moved jewels are constraints, not suggestions from the inpainting model.

### `plan_translation_edit`
- **Does**: Builds a conservative dirty-cell mask from the swept selection AABB plus a configurable
  cell halo.
- **Rationale**: Local additive overlaps need neighborhood reconciliation; exact endpoint cells alone
  are too tight.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Future editor UI | Selection delta maps to source/destination/sweep semantics | Dirty-region policy |
| Future latent inpainting CLI | Clean context is outside every dirty cell | Context filtering |
| Tests | Protected jewels move exactly once and remain mergeable | Merge ordering/semantics |

## Notes

- The future latent prior should condition dirty-cell sampling on `protected_moved`, likely through a
  small set encoder or per-cell protected-jewel summary.
- Generated destination content may otherwise collide additively with protected jewels. Training
  must expose the model to this constraint; post-hoc merging alone cannot teach reconciliation.
