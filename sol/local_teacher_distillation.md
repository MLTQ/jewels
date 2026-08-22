# `local_teacher_distillation.py`

## Purpose

Defines local fitted-teacher targets and correspondence losses for factorized irregular jewels.
The module keeps attribute matching separate from the trainer and deliberately prevents appearance
or covariance targets from moving center coordinates through correspondence weights.

## Components

### `LocalTeacherAttributes`
- **Does**: Owns sampled teacher centers, ordered log scales, principal axes, opacity, base color,
  color gradient, and the full field's active-jewel count.
- **Does**: Moves tensor fields between devices without changing the physical active count.

### `extract_local_teacher_attributes`
- **Does**: Opacity-samples canonical 22-D teacher features and decomposes covariance into ordered
  log scales plus the largest-eigenvalue axis.
- **Rationale**: The old global spread/size descriptors could match aggregate shape while assigning
  the wrong covariance and appearance to a specific content region.

### `soft_local_correspondence`
- **Does**: Selects the nearest fitted jewels and gives them normalized Gaussian distance weights.
- **Rationale**: Detached weights let local attributes supervise the matched jewel without creating
  a shortcut that moves centers toward easy colors.

### `local_teacher_attribute_losses`
- **Does**: Separately scores ordered scale, sign-invariant principal axis, optical opacity mass,
  base color, and color gradient.
- **Does**: Compensates opacity optical density for the fitted/student active-count ratio and keeps
  every component visible to experiment logs.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `distill_structural_encoder.py` | Loss dictionary has scale/axis/opacity/color/gradient | Keys or semantics |
| Fitted checkpoints | Canonical features remain center 0:3, covariance 3:9, RGB 9:12, gradient 12:21, opacity 21 | Feature layout |
| Factorized-v3 experiment | Correspondence has no gradient path into student centers | Removing detach/no-grad |
| Opacity supervision | Active count uses the canonical 2% threshold | Threshold policy |
