# Spacetime Jewel geometry figure

## Intent

This figure defines one Jewel without assuming prior Gaussian-splatting vocabulary. It shows a
tilted Gaussian support in the displayed-video volume `(u, v, t)`, three temporal cross-sections,
and the complete 22-parameter representation.

## Contract

- Time is a depth axis of the spacetime volume; the depicted motion is the changing intersection
  of a fixed-time slice with a tilted support.
- The covariance is symmetric positive definite and contributes six independent parameters.
- The color Jacobian is local RGB variation over `(u, v, t)` and contributes nine parameters.
- The figure describes a displayed 2D video volume, not a multi-view 3D scene.

## Maintenance

Update the parameter count in the figure, method equation, and prose together if the primitive
definition changes.
