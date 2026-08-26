# `test_addressed_scene_block_realizer.py`

## Purpose

Protects the corrected `(scene, address, local token)` constellation lookup used by Gate 2a7.

## Coverage

- Nearest candidates remain inside the requested semantic scene family.
- Candidate descriptors and templates are compared only at matching block addresses.
- Sampling emits the exact requested continuous Jewel count and aligned active tokens.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a7 audit | Same-address, scene-owned candidate selection | Eligibility or indexing |
