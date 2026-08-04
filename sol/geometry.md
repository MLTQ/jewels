# `geometry.py`

## Purpose

Defines the cursor-visible video volume and the first supported edit: translating selected jewels
inside an oriented parallelepiped. It deliberately keeps selection math independent of rendering
and learning.

## Components

### `Parallelepiped`
- **Does**: Represents an oriented `(u,v,t)` selection by a center and three half-edge vectors.
- **Interacts with**: `EditPlan` in `edit.py`, `GridSpec.cells_for_aabb` in `token_grid.py`.
- **Rationale**: This is the exact geometric object the final viewer presents to the user.

### `Parallelepiped.contains`
- **Does**: Solves world points into local box coordinates and tests `[-1,1]^3` containment.

### `Parallelepiped.world_aabb`
- **Does**: Returns a conservative axis-aligned bound for dirty-cell marking.
- **Rationale**: Over-marking cells is safe for local inpainting; missing an intersected cell is not.

### `translate_selected`
- **Does**: Moves feature dimensions `0:3` for selected jewels and returns the selection mask.
- **Interacts with**: Canonical 22-value feature layout from `stprim/prior/featurize.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `edit.py` | Selection and translated selection use identical basis geometry | Selection semantics |
| `tests/test_edit_inpaint.py` | Pure translation changes centers only | Feature layout |

## Notes

- Rotation and non-rigid transforms are intentionally deferred. They require covariance congruence
  transforms and a clear policy for P1 color gradients.
