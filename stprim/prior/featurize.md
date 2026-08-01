# featurize.py

## Purpose
The stage-1/stage-2 boundary: fit checkpoints <-> fixed-width training tensors. Everything the
prior learns flows through this representation, so its gauge choices ARE the model's world.

## Components

### `state_to_features(state)` / `features_to_field(feats)`
- **Does**: PrimitiveField state <-> (N, 22) features; decode returns a renderable field
- **Rationale**: geometry travels as **log-covariance** (log-Euclidean SPD coords), never
  (scale, quat). Quaternions double-cover rotations (q ≡ -q) and eigen-axes permute — raw
  params would make identical primitives distant in feature space and force the model to
  learn gauge. logSigma is unique and symmetric; decode eigendecomposes, and any (R, s)
  factorization renders identically, so eigh's sign/permutation freedom is harmless
  (det<0 flips one column to stay in SO(3) for the quat conversion).

### `rotmat_to_quat(R)`
- **Does**: batched Shepperd with per-row best-pivot selection
- **Rationale**: needed only at decode; PrimitiveField's parameterization is quat.

### `load_corpus(dir)`
- **Does**: stack all `*_w*.pt` + `.clip.npy` -> feats (S,N,22), clip (S,D), bg (S,3), shape
- **Rationale**: asserts uniform N — v0 trains fixed-size sets (the Avenue corpus is uniformly
  6471 because the densify schedule is deterministic and prune never fired). Padding/masks
  arrive with the first corpus that needs them.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `cli/train_prior.py`, `cli/sample_prior.py` | FEAT_DIM=22 layout above | any layout change invalidates trained priors + their stats |

## Notes
- Feature layout: mu(3) | logSigma-triu(6) | color(3) | color_grad(9) | logit_w(1).
- color_grad is world-frame (d color / d position), so it carries no rotation gauge.
- Round-trip fidelity is limited only by eigh numerics — verified ~1e-6 max error.
