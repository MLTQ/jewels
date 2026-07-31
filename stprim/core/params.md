# params.py

## Purpose
Parameter container for N anisotropic spacetime primitives — and the unit the stage-2
generative model will emit: a video is a set of these. (It used to be shared by an additive
and a Voronoi renderer; the Voronoi branch was measured and removed 2026-07-31, see
PROJECT.md.)

## Components

### `quat_to_rotmat(q)`
- **Does**: (...,4) quaternion -> (...,3,3) rotation matrix, normalizing internally
- **Rationale**: Normalizing inside means `quat` is an unconstrained parameter — no
  renormalization step in the training loop, no projection operator.

### `PrimitiveField`
- **Does**: holds mu, log_scale, quat, color, optional color_grad, logit_w
- **Interacts with**: `models/render.py` (via `gather`), `fit/adapt.py` (via `subset_`/`append_`)
- **Rationale**: (scale, quat) define the per-primitive Gaussian covariance. For stage 2, the
  covariance — not (scale, quat) — is the canonical object: quaternions double-cover rotations
  (q ≡ -q) and axis order is permutable, so raw parameters must be canonicalized before being
  fed to a prior or quantized into a vocabulary.

### `PrimitiveField.gather(idx)`
- **Does**: gathers params for an (M,K) index tensor after culling -> (M,K,...) tensors
- **Interacts with**: `models/render.cull_knn` produces `idx`

### `PrimitiveField.subset_` / `append_`
- **Does**: in-place prune / grow
- **Rationale**: replaces the nn.Parameter objects outright. Callers MUST rebuild the optimizer.

## Decisions
- `color_grad` (P1 linear ramp) is optional but on by default — P0 constant color needs far more
  primitives for smooth gradients.
- `log_scale` is clamped to [-8, 2] in `scales()` rather than parameterized through a softplus:
  clamping keeps gradients clean at the boundary and the range is generous.
- Quaternion over Euler angles / 6D rotation: no gimbal issues, 4 params, trivially normalized.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `models/render.py` | `gather(idx)` returns keys mu/scale/rot/color/logit_w (+color_grad if p1) | Key names, shapes |
| `fit/adapt.py` | `subset_(mask)`, `append_(dict)`, `.weights()`, `.scales()` | Signatures; optimizer invalidation contract |
| `fit/fitter.py` | parameters exposed as named attributes for per-group LRs | Attribute names |

## Notes
- Weights go through sigmoid, so `logit_w=0` -> 0.5. Default prune threshold (0.01) therefore
  prunes nothing at init — intentional, pruning should only bite after training moves weights.
- Random init is drawn on the *generator's* device and then moved to the target device. CUDA
  ops reject a CPU generator (this crashed the first GPU run), and drawing on the generator's
  device makes a given seed produce a bit-identical init on CPU and GPU — which matters for the
  canonicalization experiment, where init seeds are the independent variable.
