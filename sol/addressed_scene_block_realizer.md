# `addressed_scene_block_realizer.py`

## Purpose

Implements the intended native grammar `(scene token, block address, local block token)`. It fixes
the address-free medoid lookup that transplanted otherwise valid constellations between unrelated
regions in Gates 2a4-2a6.

## Components

### `AddressedSceneBlockRealizer`

- **Does**: Selects complete same-address training constellations by semantic scene and local token.
- **Does**: Pools the four nearest same-address candidates for likelihood and uses the nearest one
  for joint generation.
- **Does**: Preserves continuous local coordinates, exact output count, and adjustment diagnostics.

### `nearest_fields`

- **Does**: Compares the addressed token prototype only with eligible fields at that same address.
- **Rationale**: Address is part of the phrase syntax; ignoring it changes the language.

### `fit_addressed_scene_block_realizer`

- **Does**: Precomputes training-owned normalized descriptors, per-block role histograms, and
  canonically sorted complete constellations for efficient source-disjoint audits.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a7 audit | Fine 16x16x8 addresses, three semantic scenes plus pooled null | Lookup semantics |
| Future Gate 2b | Address is implicit sequence position; token remains K=1024 | Program syntax |
| Gate 0f decoder | Output is continuous centers plus three active tokens | Output schema |
| Scientific review | Only same-address training blocks may realize a phrase | Eligibility rule |
