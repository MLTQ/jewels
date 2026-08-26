# `scene_block_constellation_realizer.py`

## Purpose

Defines the first explicit two-level native Jewel utterance: one global scene token selects a
coherent family of local templates, then 256 K=1024 block tokens select complete predefined Jewel
constellations within that family.

## Components

### `SceneBlockConstellationRealizer`

- **Does**: Indexes complete medoid templates and smoothed role likelihoods by `(scene, block)`.
- **Does**: Casts all addressed joint constellations with continuous jitter and reports exact-count
  adjustment.
- **Does**: Reserves the final scene token for the prompt-blind pooled control.

### `fit_scene_block_constellation_realizer`

- **Does**: Chooses the nearest eligible medoid for every scene/block token pair.
- **Does**: Pools the four nearest eligible training blocks for likelihood while retaining one
  complete nearest constellation for generation.
- **Rationale**: The global token prevents adjacent blocks from independently selecting mutually
  incompatible style/action templates.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a5 audit | Three semantic scene tokens plus one pooled null, K=1024 blocks | Index semantics |
| Future Gate 2b | Scene token precedes time-major Morton block sequence | Token hierarchy |
| Gate 0f decoder | Output remains continuous centers plus active Jewel tokens | Output schema |
| Scientific review | Six registered training videos own each semantic template family | Data ownership |
