# `test_hierarchical_inpaint.py`

## Purpose

Protects the mask-layout and exact-clamping invariants connecting 32³ edit selection to 16³ flow
repair.

## Coverage

- A touched fine cell dirties its entire 2³ block and no unrelated block.
- Batched masks preserve canonical ordering.
- Hierarchical sampling leaves every clean coarse code bit-identical.
- Raw codes are reclamped after normalization arithmetic.
- Invalid block/grid combinations fail explicitly.
