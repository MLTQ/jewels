# `local_teacher_distillation.py`

## Purpose

Defines local fitted-teacher targets, position-only correspondence, and renderer-responsibility
moments for factorized irregular jewels. The module keeps attribute matching separate from the
trainer and deliberately prevents appearance or covariance targets from moving center coordinates.

## Components

### `LocalTeacherAttributes`
- **Does**: Owns sampled teacher centers, ordered log scales, principal axes, opacity, base color,
  color gradient, covariance/precision, and the full field's active-jewel count.
- **Does**: Moves tensor fields between devices without changing the physical active count.

### `extract_local_teacher_attributes`
- **Does**: Opacity-samples or uniformly samples active canonical 22-D teacher features and
  decomposes covariance into ordered log scales plus the largest-eigenvalue axis.
- **Rationale**: The old global spread/size descriptors could match aggregate shape while assigning
  the wrong covariance and appearance to a specific content region.

### `soft_local_correspondence`
- **Does**: Selects the nearest fitted jewels and gives them normalized Gaussian distance weights.
- **Rationale**: Detached weights let local attributes supervise the matched jewel without creating
  a shortcut that moves centers toward easy colors.

### `renderer_responsibility_targets`
- **Does**: Evaluates each sampled teacher's opacity-weighted Mahalanobis contribution at detached
  student centers under a declared finite support.
- **Does**: Produces mixture-covariance moments, optical density, locally rendered color, and the
  analytic Jacobian of normalized local color.
- **Does**: Marks queries with no sampled teacher inside the declared support before falling back to
  the minimum-Mahalanobis teacher, so sample insufficiency remains measurable.
- **Rationale**: Several overlapping fitted jewels can explain one rendered observation. Moment
  targets describe their joint local response instead of assigning one raw jewel by center distance.

### `local_teacher_attribute_losses`
- **Does**: Separately scores ordered scale, sign-invariant principal axis, optical opacity mass,
  base color, and color gradient.
- **Does**: Compensates opacity optical density for the fitted/student active-count ratio and keeps
  every component visible to experiment logs.

### `responsibility_teacher_moment_losses`
- **Does**: Scores the same five independently weighted student attributes against composited
  responsibility moments and returns effective/support counts for diagnostics.
- **Does**: Projects targets to the v3 student's declared feasible ranges (log scale `[-9,1]`,
  optical density at most `6`, RGB `[0,1]`, and RGB Jacobian `[-0.25,0.25]`).

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `distill_structural_encoder.py` | Loss dictionary has scale/axis/opacity/color/gradient | Keys or semantics |
| Fitted checkpoints | Canonical features remain center 0:3, covariance 3:9, RGB 9:12, gradient 12:21, opacity 21 | Feature layout |
| Factorized-v3 experiment | Correspondence has no gradient path into student centers | Removing detach/no-grad |
| Opacity supervision | Active count uses the canonical 2% threshold | Threshold policy |
| Responsibility experiment | Active-uniform teacher sampling applies opacity exactly once in contribution logits | Sampling/weight semantics |
| Renderer consistency | Contribution logits use the canonical covariance precision and five-sigma-style support | Feature/support semantics |
