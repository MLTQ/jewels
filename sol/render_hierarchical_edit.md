# `render_hierarchical_edit.py`

## Purpose

Runs the first end-to-end editor proof: select jewels in a `(u,v,t)` parallelepiped, translate them,
regenerate only the hierarchy blocks touched by the swept volume, merge protected moved jewels, and
render the result.

## Pipeline

1. Build a fine-grid `EditPlan` from selection, translation, sweep, and halo.
2. Conservatively map the fine dirty cells onto 2³ PCA blocks.
3. Clamp all clean 16³ codes while the axial prior samples dirty codes.
4. Reapply raw clean codes after de-normalization to eliminate floating-point round-trip drift.
5. Decode the repaired hierarchy, retain generated jewels only in affected blocks, and merge the
   untouched context plus exactly translated protected jewels.
6. Render target, hierarchy baseline, direct move, and locally repaired move side by side.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Research audit | Reports exact clean-code and clean-fine errors | Manifest fields |
| Future editor | Moved jewels survive as protected constraints | Merge order/policy |
| Renderer | Output remains a variable-size jewel set in canonical 22-D feature format | Feature layout |

## Limitations

- The prior has not yet seen masked-repair training examples.
- Protected jewels are merged after generation but are not yet encoded into the prior condition, so
  collision reconciliation is not learned.
- The command currently exposes an axis-aligned selection; the underlying geometry supports an
  oriented parallelepiped.
