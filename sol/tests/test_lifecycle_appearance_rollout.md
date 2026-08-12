# `test_lifecycle_appearance_rollout.py`

## Purpose

Protects the autonomous ownership split across initial generation and multiple continuation
windows.

## Components

### `LifecycleAppearanceRolloutTests`

- **Does**: Verifies the interleaved frozen branch reproduces the existing standalone rollout,
  both fields retain identical counts and stable IDs, every lifecycle coordinate is exact, and a
  nonzero static-detail residual survives three strides while every omitted coordinate remains
  exact through topology projection and global covariance conversion. Mixed cell gates also retain
  zero-strength ranks and report their realized active fraction.
- **Interacts with**: `lifecycle_appearance_rollout.py` and `scaffold_mark_rollout.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Four-class experiment | Frozen base is a matched control rather than a new sampling realization | RNG sequence |
| Persistent editor | Candidate changes never reorder or delete base-owned jewel rows | Append policy |
