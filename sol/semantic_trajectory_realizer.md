# `semantic_trajectory_realizer.py`

## Purpose

Composes two coherent Jewel programs through a class-discriminative, density-balanced moving tube.
It corrects the donor-disagreement mask diagnosed by Gate 2a9 without changing the physical Jewel
language.

## Components

### `SemanticTrajectoryRealizer`

- **Does**: Stores one training-only semantic trajectory per scene token, selects two distinct
  coherent donors, and chooses a fixed-candidate radius that balances inserted foreground against
  removed background.
- **Does**: Applies the correct-scene trajectory even when the foreground donor is intentionally
  drawn from a wrong scene, localizing semantic ownership to the persistent tube.
- **Does**: Exposes `sample_from_donors` so a prompt-only speaker can emit explicit foreground and
  background tokens without a target-derived block program.
- **Does**: Exposes `sample_rank_balanced_from_donors`, which takes the closest half of a foreground
  donor and farthest half of a background donor relative to the semantic path. This gives a
  prompt-only program exact count and equal ownership without assuming two decompositions have
  equal density at one geometric radius.

### `fit_semantic_trajectory_realizer`

- **Does**: Builds per-scene saliency from squared differences between same-scene and other-scene
  mean addressed descriptors, with a centered spatial prior and temporal `[1,2,1]` smoothing.
- **Rationale**: The path encodes where the prompt class differs, not merely where two arbitrary
  source samples disagree.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Gate 2a10 audit | 132 radius candidates in [0.45, 1.10] and 20% donor floor | Balance rule |
| Leakage audit | Scene paths use training descriptors only | Path ownership |
| Future speaker | Persistent scene/track state owns cross-block and cross-time dependencies | Grammar shape |
