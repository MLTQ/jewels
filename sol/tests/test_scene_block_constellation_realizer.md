# `test_scene_block_constellation_realizer.py`

## Purpose

Protects the two-level scene-token/block-token realization contract of Gate 2a5.

## Coverage

- Semantic scene families plus a pooled null token are created.
- A scene/block program casts the exact requested continuous Jewel count.
- Scene ownership is preserved in generation diagnostics.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a5 audit | Dense scene token IDs and one null scene | Scene indexing |
